"""Scene affinity, masked scene attention, and scene grouping.

The 16 register tokens already in VGGT-Omega are the scene tokens. This module
does not add a second token set. It reads those tokens after the 24 aggregator
blocks, predicts which frames share a scene, and mixes scene tokens only inside
a scene.

The attention residual is zero at initialization, so a single-scene input matches
the original model until this block is trained.
"""

import torch
import torch.nn as nn


NUM_SCENE_TOKENS = 16


class AffinityHead(nn.Module):
    """Pairwise scene affinity from the mean of each frame's scene tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pair = nn.Linear(dim * 2, 1)

    def forward(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        """Return affinity of shape (B, N, N), symmetric, with a 1 diagonal."""
        if scene_tokens.ndim != 4:
            raise ValueError(f"Expected scene tokens (B, N, 16, C), got {tuple(scene_tokens.shape)}")
        descriptors = scene_tokens.mean(dim=2)
        left = descriptors[:, :, None, :].expand(-1, -1, descriptors.shape[1], -1)
        right = descriptors[:, None, :, :].expand(-1, descriptors.shape[1], -1, -1)
        logits = self.pair(torch.cat([left, right], dim=-1)).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        affinity = torch.sigmoid(logits)
        eye = torch.eye(affinity.shape[-1], device=affinity.device, dtype=affinity.dtype)
        return affinity * (1.0 - eye) + eye


class SceneAttention(nn.Module):
    """One attention block over the 16 scene tokens of every frame.

    `bias` is (B, N, N) and is expanded so each image pair controls a 16x16 block.
    The output projection and the MLP output start at zero, so the block is a no-op
    until it is trained.
    """

    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, scene_tokens: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_tokens, dim = scene_tokens.shape
        if num_tokens != NUM_SCENE_TOKENS:
            raise ValueError(f"Expected {NUM_SCENE_TOKENS} scene tokens, got {num_tokens}")
        if bias.shape != (batch_size, num_frames, num_frames):
            raise ValueError(
                f"Expected bias {(batch_size, num_frames, num_frames)}, got {tuple(bias.shape)}"
            )

        tokens = scene_tokens.reshape(batch_size, num_frames * num_tokens, dim)
        token_bias = bias.repeat_interleave(num_tokens, dim=1).repeat_interleave(num_tokens, dim=2)
        updated = tokens + self._attend(self.norm1(tokens), token_bias)
        updated = updated + self.mlp(self.norm2(updated))
        return updated.reshape(batch_size, num_frames, num_tokens, dim)

    def _attend(self, tokens: torch.Tensor, token_bias: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores + token_bias[:, None, :, :]
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        return self.proj(mixed)


class MultiScene(nn.Module):
    """Affinity head plus masked scene attention on the final aggregator tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        threshold: float = 0.5,
        bias_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.bias_scale = float(bias_scale)
        self.affinity = AffinityHead(dim)
        self.scene_attention = SceneAttention(dim, num_heads=num_heads)

    def forward(
        self,
        tokens: torch.Tensor,
        patch_token_start: int,
        hard_mask: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Update scene tokens and return affinity plus group ids.

        `tokens` is (B, N, T, C) with the camera token at index 0 and the 16 scene
        tokens immediately after it. Other tokens are left unchanged.
        """
        num_scene_tokens = patch_token_start - 1
        if num_scene_tokens != NUM_SCENE_TOKENS:
            raise ValueError(
                f"Expected {NUM_SCENE_TOKENS} scene tokens, got {num_scene_tokens}. "
                "They are the existing register tokens."
            )

        device_type = tokens.device.type if tokens.device.type in ("cuda", "mps", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            scene_tokens = tokens[:, :, 1:patch_token_start].float()
            affinity = self.affinity(scene_tokens)
            bias = build_attention_bias(
                affinity.detach(),
                hard_mask=hard_mask,
                threshold=self.threshold,
                bias_scale=self.bias_scale,
            )
            updated = self.scene_attention(scene_tokens, bias)
            affinity_output = self.affinity(updated)

        camera_tokens = tokens[:, :, :1]
        patch_tokens = tokens[:, :, patch_token_start:]
        tokens = torch.cat([camera_tokens, updated.to(dtype=tokens.dtype), patch_tokens], dim=2)
        group_id = scene_groups(affinity.detach(), self.threshold)
        return tokens, {
            "affinity": affinity,
            "affinity_output": affinity_output,
            "group_id": group_id,
        }


def predict_pose_by_scene(
    camera_head: nn.Module,
    tokens: torch.Tensor,
    patch_token_start: int,
    group_id: torch.Tensor,
) -> torch.Tensor:
    """Run the camera head once per scene so its attention stays inside the group.

    `tokens` is (B, N, T, C). `group_id` is (B, N). Frames keep their original order
    inside a group, so the first frame of a scene stays the anchor.
    """
    if group_id.ndim != 2:
        raise ValueError(f"Expected group ids (B, N), got {tuple(group_id.shape)}")
    batch_size, num_frames = group_id.shape
    if tokens.shape[0] != batch_size or tokens.shape[1] != num_frames:
        raise ValueError(
            f"Tokens {(tokens.shape[0], tokens.shape[1])} do not match group ids {tuple(group_id.shape)}"
        )

    rows = [
        _pose_for_one_bag(camera_head, tokens[batch_index], patch_token_start, group_id[batch_index])
        for batch_index in range(batch_size)
    ]
    return torch.stack(rows, dim=0)


def _pose_for_one_bag(
    camera_head: nn.Module,
    frame_tokens: torch.Tensor,
    patch_token_start: int,
    group_id: torch.Tensor,
) -> torch.Tensor:
    num_frames = frame_tokens.shape[0]
    pose = None
    for scene in torch.unique(group_id, sorted=True):
        index = torch.nonzero(group_id == scene, as_tuple=False).flatten()
        scene_tokens = frame_tokens.index_select(0, index.to(frame_tokens.device)).unsqueeze(0)
        predicted = camera_head([scene_tokens], patch_token_start)
        index = index.to(predicted.device)
        if predicted.shape[1] != index.shape[0]:
            raise ValueError(
                f"Camera head returned {predicted.shape[1]} frames for a scene of {index.shape[0]}"
            )
        if pose is None:
            pose = predicted.new_zeros(num_frames, predicted.shape[-1])
        pose = pose.index_copy(0, index, predicted[0])
    if pose is None:
        raise ValueError("Cannot predict poses for an empty frame list")
    return pose


def select_camera_groups(
    predicted_groups: torch.Tensor,
    scene_id: torch.Tensor | None,
    camera_groups: str,
) -> torch.Tensor:
    """Choose which grouping the camera head uses.

    `gt` follows the labels while training. Inference has no labels, so it falls
    back to the predicted groups. `predicted` always uses the affinity groups.
    """
    if camera_groups not in ("gt", "predicted"):
        raise ValueError(f"camera_groups must be 'gt' or 'predicted', got {camera_groups!r}")
    if camera_groups == "gt" and scene_id is not None:
        if scene_id.shape != predicted_groups.shape:
            raise ValueError(
                f"scene_id {tuple(scene_id.shape)} does not match groups {tuple(predicted_groups.shape)}"
            )
        return scene_id
    return predicted_groups


def build_attention_bias(
    affinity: torch.Tensor,
    hard_mask: bool,
    threshold: float,
    bias_scale: float,
) -> torch.Tensor:
    """Turn affinity into an additive attention bias of shape (B, N, N).

    Training uses a soft bias, log(affinity), so the mask has a gradient when a
    downstream loss asks for one. Inference uses a hard 0 / very-negative mask.
    The diagonal stays open because affinity is forced to 1 there.
    """
    if hard_mask:
        blocked = affinity < threshold
        bias = torch.zeros_like(affinity)
        return bias.masked_fill(blocked, torch.finfo(affinity.dtype).min)
    return bias_scale * torch.log(affinity.clamp(min=1e-6))


@torch.no_grad()
def scene_groups(affinity: torch.Tensor, threshold: float) -> torch.Tensor:
    """Connected components of affinity >= threshold.

    Returns group ids (B, N), starting at 0 in each row. A chain of pairs becomes
    one group even when the direct pair is below the threshold.
    """
    same = affinity >= threshold
    batch_size, num_frames, _ = same.shape
    groups = torch.arange(num_frames, device=affinity.device).view(1, -1).expand(batch_size, -1).clone()
    large = num_frames
    for _ in range(max(num_frames - 1, 0)):
        candidate = groups[:, None, :].expand(batch_size, num_frames, num_frames).clone()
        candidate = candidate.masked_fill(~same, large)
        groups = torch.minimum(groups, candidate.amin(dim=2))

    remapped = torch.empty_like(groups)
    for batch_idx in range(batch_size):
        mapping: dict[int, int] = {}
        row = []
        for frame_idx in range(num_frames):
            root = int(groups[batch_idx, frame_idx])
            if root not in mapping:
                mapping[root] = len(mapping)
            row.append(mapping[root])
        remapped[batch_idx] = torch.tensor(row, dtype=groups.dtype, device=affinity.device)
    return remapped
