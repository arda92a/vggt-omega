"""Scene router: DINOv3 features per frame, a small transformer over the frames of a bag, a pairwise affinity matrix.

    frame -> DINOv3 (CLS + mean patch) -> MLP -> bag transformer (frames see each other) -> embedding e_i
    affinity logit(i, j) = scale * cos(e_i, e_j) + bias        sigmoid of it: P(i and j are the same sequence)

The DINOv3 weights are the official .pth files, read from weights_dir by arch, for example
dino_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth for arch vits16.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Official DINOv3 LVD-1689M backbones (dinov3/hub/backbones.py). The "plus" sizes use SwiGLU and are not listed.
ARCHS = {
    "vits16": dict(embed_dim=384, depth=12, num_heads=6),
    "vitb16": dict(embed_dim=768, depth=12, num_heads=12),
    "vitl16": dict(embed_dim=1024, depth=24, num_heads=16),
}
COMMON = dict(
    img_size=224, patch_size=16, pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_dtype="fp32", ffn_ratio=4, qkv_bias=True, drop_path_rate=0.0, layerscale_init=1.0e-5,
    norm_layer="layernormbf16", ffn_layer="mlp", ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
    # The official models set pos_embed_rope_rescale_coords=2, a train-time augmentation of positions only.
    # It is left off so fine-tuning sees the same positions as inference.
)


def find_weights(weights_dir: str, arch: str) -> Path:
    """The one dinov3_<arch>_pretrain*.pth file in weights_dir."""
    folder = Path(weights_dir).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"weights_dir {folder} does not exist")
    found = sorted(folder.glob(f"dinov3_{arch}_pretrain*.pth"))
    if not found:
        present = ", ".join(p.name for p in folder.glob("*.pth")) or "no .pth files"
        raise FileNotFoundError(f"no DINOv3 {arch} weights (dinov3_{arch}_pretrain*.pth) in {folder}, found: {present}")
    if len(found) > 1:
        raise FileNotFoundError(f"several DINOv3 {arch} weights in {folder}: {', '.join(p.name for p in found)}")
    return found[0]


def build_dinov3(arch: str, weights_dir: str) -> DinoVisionTransformer:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {list(ARCHS)}, got {arch!r}")
    path = find_weights(weights_dir, arch)
    logging.getLogger("dinov3").setLevel(logging.WARNING)  # quiet the per-build RoPE messages
    model = DinoVisionTransformer(**COMMON, **ARCHS[arch])
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if "local_cls_norm" not in k}  # unused head of some checkpoints
    model.load_state_dict(state, strict=True)
    logging.getLogger("scene_router").info(f"DINOv3 {arch} weights: {path}")
    return model


def set_trainable(backbone: DinoVisionTransformer, train_blocks: int) -> None:
    """Freeze the backbone except its last `train_blocks` blocks and final norm (0 freezes all of it)."""
    backbone.requires_grad_(False)
    if train_blocks > 0:
        for block in backbone.blocks[-train_blocks:]:
            block.requires_grad_(True)
        backbone.norm.requires_grad_(True)


class SceneRouter(nn.Module):
    def __init__(self, backbone: DinoVisionTransformer, dim: int = 256, context_layers: int = 2, heads: int = 8):
        super().__init__()
        self.backbone = backbone
        width = backbone.embed_dim
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.project = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, dim), nn.GELU(), nn.Linear(dim, dim))
        self.context = None
        if context_layers > 0:
            layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, context_layers, enable_nested_tensor=False)
        self.embed = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def frame_features(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 images (M, 3, H, W) -> (M, 2 * width): CLS token and mean patch token."""
        x = (images.float() / 255 - self.mean) / self.std
        out = self.backbone.forward_features(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], dim=-1).float()

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """images (B, N, 3, H, W) with mask (B, N) of real frames -> affinity logits (B, N, N)."""
        batch, frames = mask.shape
        features = self.project(self.frame_features(images[mask]))  # padded frames never reach the backbone
        tokens = features.new_zeros(batch, frames, features.shape[-1])
        tokens[mask] = features
        if self.context is not None:
            tokens = self.context(tokens, src_key_padding_mask=~mask)
        embeddings = F.normalize(self.embed(tokens), dim=-1)
        return self.scale * embeddings @ embeddings.transpose(1, 2) + self.bias


def build_router(cfg: dict) -> SceneRouter:
    backbone = build_dinov3(cfg["arch"], cfg["weights_dir"])
    set_trainable(backbone, cfg["train_blocks"])
    return SceneRouter(backbone, dim=cfg["dim"], context_layers=cfg["context_layers"])
