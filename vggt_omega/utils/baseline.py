"""Training-free scene grouping baseline: cosine similarity of mean patch tokens from an early backbone block."""

import contextlib

import torch
import torch.nn.functional as F

from vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten


@torch.no_grad()
def patch_feature_affinity(aggregator: Aggregator, images: torch.Tensor, layer: int = 4) -> torch.Tensor:
    """Cosine affinity (N, N) between frames, from the mean patch token after block `layer` (1-based).

    `images` is (N, 3, H, W) or (1, N, 3, H, W) in [0, 1]. Block 4 scored best on retail bags
    (ARI 0.885 with a threshold of about 0.90) without any training; see training/baselines.py.
    """
    features = patch_features(aggregator, images, layer)
    return features @ features.T


@torch.no_grad()
def patch_features(aggregator: Aggregator, images: torch.Tensor, layer: int = 4) -> torch.Tensor:
    """Unit-length mean patch token (N, C) of every frame after block `layer`, on the CPU."""
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
        features = tokens.reshape(num_frames, per_frame, dim)[:, aggregator.patch_token_start :].float().mean(1)
    features = F.normalize(features, dim=-1)
    return features.cpu()
