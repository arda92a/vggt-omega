"""Learned patch matching on frozen VGGT-Omega features: the trainable version of the best training-free method.

The training-free method (global/match70-c@4 in training/evaluate.py, ARI 0.943 on bags_eval) does, for frames A, B:
    patches after VGGT-Omega block 4 -> subtract the bag's mean patch -> for every patch of A its best cosine in B
    -> share of patches whose best match is >= 0.70 -> same scene when the share is >= 0.20

Here the fixed parts become learnable, and everything starts at the training-free values, so step 0 already
behaves like that method:
    adapter       x + up(down(x)) on every patch token, rank `adapter_rank`; up starts at zero, so it starts as identity
    soft match    sigmoid((best - tau) / temperature) instead of best >= 0.70; tau and temperature are learned
    two-hop       frames of one long stitch request often share no patches (frame 1 and frame 10). If A matches K
                  and K matches B, that is evidence for A and B: max over K of min(share(A, K), share(K, B))
    pair logit    scale * (share + two_hop_weight * two_hop - cutoff); cutoff starts at 0.20, the weight at 0

VGGT-Omega itself stays frozen and only its first `layer` blocks are kept in memory.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.baseline import patch_tokens


def load_aggregator(weights: str, layer: int) -> nn.Module:
    """The frozen VGGT-Omega backbone (aggregator), cut after block `layer`."""
    state = torch.load(weights, map_location="cpu", weights_only=False)
    state = state.get("model", state) if isinstance(state, dict) else state
    model = VGGTOmega(enable_camera=False, enable_depth=False)
    result = model.load_state_dict(state, strict=False)
    if any(key.startswith("aggregator.") for key in result.missing_keys):
        raise SystemExit(f"{weights} is missing aggregator weights")
    aggregator = model.aggregator
    aggregator.frame_blocks = nn.ModuleList(list(aggregator.frame_blocks)[:layer])
    aggregator.inter_frame_blocks = nn.ModuleList(list(aggregator.inter_frame_blocks)[:layer])
    aggregator.depth = layer
    aggregator.requires_grad_(False)
    return aggregator.eval()


class VGGTPatchMatcher(nn.Module):
    def __init__(self, aggregator: nn.Module, layer: int = 4, adapter_rank: int = 64, patch_threshold: float = 0.70,
                 temperature: float = 0.03, share_threshold: float = 0.20, scale: float = 20.0, two_hop: bool = True):
        super().__init__()
        self.aggregator = aggregator
        self.layer = layer
        width = aggregator.camera_token.shape[-1]
        self.down = nn.Linear(width, adapter_rank, bias=False)
        self.up = nn.Linear(adapter_rank, width, bias=False)
        nn.init.zeros_(self.up.weight)  # identity at the start
        self.tau = nn.Parameter(torch.tensor(float(patch_threshold)))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        self.cutoff = nn.Parameter(torch.tensor(float(share_threshold)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        self.two_hop_weight = nn.Parameter(torch.tensor(0.0)) if two_hop else None

    def train(self, mode: bool = True):
        super().train(mode)
        self.aggregator.eval()  # frozen backbone: always in inference mode
        return self

    def checkpoint_state(self) -> dict:
        """Only the learned weights: the frozen VGGT-Omega part is rebuilt from its own weights file."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("aggregator.")}

    def param_groups(self, cfg: dict) -> list[dict]:
        scalars = [self.tau, self.log_temperature, self.cutoff, self.log_scale]
        if self.two_hop_weight is not None:
            scalars.append(self.two_hop_weight)
        return [
            {"params": [self.down.weight, self.up.weight], "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
            {"params": scalars, "lr": cfg["lr"], "weight_decay": 0.0},  # decay would pull tau and cutoff to 0
        ]

    def describe(self) -> str:
        return (f"VGGT-Omega block {self.layer} (frozen) + patch matching head, adapter rank {self.down.out_features}, "
                f"tau {self.tau.item():.3f}, temperature {self.log_temperature.exp().item():.3f}, "
                f"cutoff {self.cutoff.item():.3f}"
                + (f", two-hop weight {self.two_hop_weight.item():.3f}" if self.two_hop_weight is not None else ""))

    def match_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Patch tokens (N, P, C) of one bag -> pair logits (N, N)."""
        with torch.no_grad():
            tokens = tokens - tokens.mean((0, 1), keepdim=True)  # centering over the bag, as in the evaluation
        x = F.normalize(tokens + self.up(self.down(tokens)), dim=-1)
        frames = torch.arange(len(x), device=x.device)[:, None]
        temperature = self.log_temperature.exp()
        share = []
        for i in range(len(x)):
            with torch.no_grad():  # which patch of each frame matches best: no gradient needed to find it
                index = torch.einsum("pc,jqc->jpq", x[i], x).argmax(-1)  # (N, P)
            best = (x[i][None] * x[frames, index]).sum(-1)  # (N, P) the best cosine, now with gradient
            share.append(torch.sigmoid((best - self.tau) / temperature).mean(-1))
        share = torch.stack(share)
        share = 0.5 * (share + share.T)
        score = share
        if self.two_hop_weight is not None and len(share) > 2:
            off = share * (1 - torch.eye(len(share), device=share.device))  # a frame is not its own middle frame
            two_hop = torch.minimum(off[:, :, None], off[None, :, :]).amax(1)  # [i, k, j] -> best middle frame k
            score = share + self.two_hop_weight * two_hop
        return self.log_scale.exp() * (score - self.cutoff)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """uint8 images (B, N, 3, H, W) with mask (B, N) of real frames -> pair logits (B, N, N)."""
        batch, frames = mask.shape
        logits = images.new_zeros(batch, frames, frames, dtype=torch.float32)
        for b in range(batch):
            n = int(mask[b].sum())
            with torch.no_grad():
                tokens = patch_tokens(self.aggregator, images[b, :n].float() / 255, self.layer)
            with torch.autocast(images.device.type, enabled=False):  # cosines near the threshold need float32
                logits[b, :n, :n] = self.match_logits(tokens.float())
        return logits


def build_matcher(cfg: dict) -> VGGTPatchMatcher:
    aggregator = load_aggregator(cfg["vggt_weights"], cfg["layer"])
    return VGGTPatchMatcher(aggregator, layer=cfg["layer"], adapter_rank=cfg["adapter_rank"],
                            patch_threshold=cfg["patch_threshold"], temperature=cfg["temperature"],
                            share_threshold=cfg["share_threshold"], two_hop=cfg.get("two_hop", True))
