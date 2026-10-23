"""Training-free scene grouping from early VGGT-Omega backbone features.

Two methods:
  patch_feature_affinity  cosine of the mean patch token of each frame (one vector per frame)
  patch_match_affinity    patch matching: every patch of frame A is compared with every patch of frame B, and the
                          score is the share of patches that find a close match. Best on retail bags_eval:
                          block 4, centering, patch threshold 0.70, frames grouped at a share of 0.20
                          (ARI 0.943, 78% of bags perfect). See training/evaluate.py and training/baselines.py.
"""

import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten

# Settings the patch matching was evaluated with on bags_eval.
MATCH_HEIGHT, MATCH_WIDTH = 448, 256
MATCH_LAYER = 4
MATCH_PATCH_THRESHOLD = 0.70
MATCH_GROUP_THRESHOLD = 0.20


def load_frames(paths: list[str], height: int = MATCH_HEIGHT, width: int = MATCH_WIDTH) -> torch.Tensor:
    """Frames as (N, 3, H, W) in [0, 1], centre-cropped to the H:W aspect and resized, the way they were evaluated.

    If most frames are landscape, height and width are swapped so landscape frames are not cut down to a portrait strip.
    """
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    if sum(w > h for w, h in sizes) > len(sizes) / 2:
        height, width = width, height
    frames = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            w, h = image.size
            if w / h > width / height:
                new_w = int(round(h * width / height))
                image = image.crop(((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h))
            else:
                new_h = int(round(w * height / width))
                image = image.crop((0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h))
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
    return torch.stack(frames).float() / 255


@torch.no_grad()
def patch_tokens(aggregator: Aggregator, images: torch.Tensor, layer: int = 4) -> torch.Tensor:
    """All patch tokens (N, P, C) of every frame after the global-attention step of block `layer` (1-based)."""
    if images.ndim == 5:
        images = images[0]
    if not 1 <= layer <= aggregator.depth:
        raise ValueError(f"layer must be in [1, {aggregator.depth}], got {layer}")
    num_frames, _, height, width = images.shape

    context = contextlib.nullcontext()
    if images.device.type == "cuda":
        context = torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    with context:
        x = ((images[None] - aggregator._resnet_mean) / aggregator._resnet_std).view(num_frames, 3, height, width)
        camera = slice_expand_and_flatten(aggregator.camera_token, 1, num_frames)
        register = slice_expand_and_flatten(aggregator.register_token, 1, num_frames)
        patch = aggregator.patch_embed(x)
        if isinstance(patch, dict):
            patch = patch["x_norm_patchtokens"]
        tokens = torch.cat([camera, register, patch], dim=1)
        _, per_frame, dim = tokens.shape
        sin, cos = aggregator.rope_embed(H=height // aggregator.patch_size, W=width // aggregator.patch_size)
        rope = (sin.to(images.device, torch.float32), cos.to(images.device, torch.float32))
        for index in range(layer):
            tokens, _ = aggregator._run_frame_block(tokens, 1, num_frames, per_frame, dim, index, rope)
            tokens = aggregator._run_inter_frame_attention_block(
                tokens, 1, num_frames, per_frame, dim, index, aggregator.inter_frame_attention_types[index], None
            )
    return tokens.reshape(num_frames, per_frame, dim)[:, aggregator.patch_token_start :].float()


@torch.no_grad()
def patch_features(aggregator: Aggregator, images: torch.Tensor, layer: int = 4) -> torch.Tensor:
    """Unit-length mean patch token (N, C) of every frame after block `layer`, on the CPU."""
    return F.normalize(patch_tokens(aggregator, images, layer).mean(1), dim=-1).cpu()


@torch.no_grad()
def patch_feature_affinity(aggregator: Aggregator, images: torch.Tensor, layer: int = 4) -> torch.Tensor:
    """Cosine affinity (N, N) between frames, from the mean patch token after block `layer` (1-based).

    `images` is (N, 3, H, W) or (1, N, 3, H, W) in [0, 1]. Block 4 scored best on retail bags
    (ARI 0.885 with a threshold of about 0.90) without any training; see training/baselines.py.
    """
    features = patch_features(aggregator, images, layer)
    return features @ features.T


@torch.no_grad()
def patch_match_affinity(
    aggregator: Aggregator, images: torch.Tensor, layer: int = MATCH_LAYER,
    patch_threshold: float = MATCH_PATCH_THRESHOLD, centered: bool = True,
) -> torch.Tensor:
    """Share of matched patches (N, N) between every two frames, symmetric, in [0, 1], on the CPU.

    A patch of frame A counts as matched in frame B when its most similar patch of B has cosine >= patch_threshold.
    The score of A and B is the mean of that share from A to B and from B to A. With centered=True the mean patch of
    all frames is subtracted first, which removes what every shelf photo shares.
    """
    patches = patch_tokens(aggregator, images, layer)
    if centered:
        patches = patches - patches.mean((0, 1), keepdim=True)
    patches = F.normalize(patches, dim=-1)
    share = torch.empty(patches.shape[0], patches.shape[0], device=patches.device)
    for i, frame in enumerate(patches):
        best = torch.einsum("pc,jqc->jpq", frame, patches).amax(-1)  # (N, P): best match of each patch of i in j
        share[i] = (best >= patch_threshold).float().mean(-1)
    return (0.5 * (share + share.T)).cpu()
