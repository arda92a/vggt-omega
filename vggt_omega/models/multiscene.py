"""Scene affinity, scene routing, and masked scene attention.

The 16 register tokens already in VGGT-Omega are the scene tokens. Nothing new is
added to the token sequence. The module does four things:

1. `route`: a first affinity head reads the scene tokens of a middle aggregator layer
   and clusters the frames. The aggregator then runs the remaining inter-frame blocks
   inside each cluster only, so later layers cannot mix scenes.
2. `forward`: after the last block a second affinity head predicts the pairwise affinity
   again, and one scene-attention block mixes the scene tokens inside each group.
3. Groups come from average-linkage clustering of the affinity, not from connected
   components, so one wrong pair cannot merge two scenes.
4. Both heads start at "everything is one scene", and the attention residual starts at
   zero, so a single-scene input reproduces the original model until training moves it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_SCENE_TOKENS = 16
LINKAGES = ("average", "single", "complete")
_INITIAL_SAME_SCENE_LOGIT = 4.0


class AffinityHead(nn.Module):
    """Pairwise same-scene logit from the mean scene token of each frame.

    The pair feature is [|d_i - d_j|, d_i * d_j]. A linear layer on [d_i ; d_j] would
    reduce to s_i + s_j and could not compare two frames at all.
    """

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.embed = nn.Linear(dim, hidden)
        self.pair = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.pair[-1].weight, std=1e-2)
        nn.init.constant_(self.pair[-1].bias, _INITIAL_SAME_SCENE_LOGIT)

    def forward(self, scene_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (affinity, logits), both (B, N, N). Affinity has a 1 diagonal."""
        if scene_tokens.ndim != 4:
            raise ValueError(f"Expected scene tokens (B, N, 16, C), got {tuple(scene_tokens.shape)}")
        descriptor = self.embed(self.norm(scene_tokens.float().mean(dim=2)))
        difference = (descriptor[:, :, None, :] - descriptor[:, None, :, :]).abs()
        product = descriptor[:, :, None, :] * descriptor[:, None, :, :]
        logits = self.pair(torch.cat([difference, product], dim=-1)).squeeze(-1)
        affinity = torch.sigmoid(logits)
        eye = torch.eye(affinity.shape[-1], device=affinity.device, dtype=torch.bool)
        return affinity.masked_fill(eye, 1.0), logits


class SceneAttention(nn.Module):
    """One attention block over the 16 scene tokens of every frame.

    Tokens attend only to tokens of frames in the same group. The output projection and
    the MLP output start at zero, so the block is a no-op until it is trained.
    """

    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        for layer in (self.proj, self.mlp[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, scene_tokens: torch.Tensor, same_group: torch.Tensor) -> torch.Tensor:
        """`scene_tokens` is (B, N, 16, C). `same_group` is a bool (B, N, N) with a True diagonal."""
        batch_size, num_frames, num_tokens, dim = scene_tokens.shape
        if num_tokens != NUM_SCENE_TOKENS:
            raise ValueError(f"Expected {NUM_SCENE_TOKENS} scene tokens, got {num_tokens}")
        if same_group.shape != (batch_size, num_frames, num_frames):
            raise ValueError(
                f"Expected same_group {(batch_size, num_frames, num_frames)}, got {tuple(same_group.shape)}"
            )

        tokens = scene_tokens.reshape(batch_size, num_frames * num_tokens, dim)
        mask = same_group.repeat_interleave(num_tokens, dim=1).repeat_interleave(num_tokens, dim=2)
        updated = tokens + self._attend(self.norm1(tokens), mask[:, None])
        updated = updated + self.mlp(self.norm2(updated))
        return updated.reshape(batch_size, num_frames, num_tokens, dim)

    def _attend(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        mixed = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return self.proj(mixed.transpose(1, 2).reshape(batch_size, num_tokens, dim))


class MultiScene(nn.Module):
    """Route affinity, final affinity, and masked scene attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        threshold: float = 0.5,
        linkage: str = "average",
        route_layer: int = 11,
    ) -> None:
        super().__init__()
        if linkage not in LINKAGES:
            raise ValueError(f"linkage must be one of {LINKAGES}, got {linkage!r}")
        self.threshold = float(threshold)
        self.linkage = linkage
        self.route_layer = int(route_layer)
        self.route_affinity = AffinityHead(dim)
        self.affinity = AffinityHead(dim)
        self.scene_attention = SceneAttention(dim, num_heads=num_heads)

    def route(self, scene_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Cluster frames from the scene tokens of the route layer. Input is (B, N, 16, C)."""
        with torch.autocast(device_type=_device_type(scene_tokens), enabled=False):
            affinity, logits = self.route_affinity(scene_tokens)
        return {
            "route_affinity": affinity,
            "route_affinity_logits": logits,
            "route_group_id": scene_groups(affinity.detach(), self.threshold, self.linkage),
        }

    def forward(
        self,
        tokens: torch.Tensor,
        patch_token_start: int,
        scene_id: torch.Tensor | None = None,
        use_gt: bool = False,
        isolate: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict final affinity and mix scene tokens inside each group.

        `tokens` is (B, N, T, C) with the camera token at index 0 and the 16 scene tokens
        right after it. Other tokens are returned unchanged. `isolate=False` puts every
        frame in one group, which is the no-isolation baseline.
        """
        if patch_token_start - 1 != NUM_SCENE_TOKENS:
            raise ValueError(
                f"Expected {NUM_SCENE_TOKENS} scene tokens, got {patch_token_start - 1}. "
                "They are the existing register tokens."
            )

        with torch.autocast(device_type=_device_type(tokens), enabled=False):
            scene_tokens = tokens[:, :, 1:patch_token_start].float()
            affinity, logits = self.affinity(scene_tokens)
            predicted = scene_groups(affinity.detach(), self.threshold, self.linkage)
            if isolate:
                used = choose_groups(predicted, scene_id, use_gt)
            else:
                used = torch.zeros_like(predicted)
            same_group = used[:, :, None] == used[:, None, :]
            updated = self.scene_attention(scene_tokens, same_group)
            output_affinity, output_logits = self.affinity(updated)

        tokens = torch.cat(
            [tokens[:, :, :1], updated.to(dtype=tokens.dtype), tokens[:, :, patch_token_start:]],
            dim=2,
        )
        return tokens, {
            "affinity": affinity,
            "affinity_logits": logits,
            "affinity_output": output_affinity,
            "affinity_output_logits": output_logits,
            "group_id": predicted,
            "used_group_id": used,
        }


def choose_groups(predicted: torch.Tensor, scene_id: torch.Tensor | None, use_gt: bool) -> torch.Tensor:
    """Ground-truth groups when `use_gt` and labels exist, else the predicted groups."""
    if not use_gt or scene_id is None:
        return predicted
    if scene_id.shape != predicted.shape:
        raise ValueError(f"scene_id {tuple(scene_id.shape)} does not match groups {tuple(predicted.shape)}")
    return scene_id


def predict_pose_by_scene(
    camera_head: nn.Module,
    tokens: torch.Tensor,
    patch_token_start: int,
    group_id: torch.Tensor,
) -> torch.Tensor:
    """Run the camera head once per group so its attention stays inside the scene.

    `tokens` is (B, N, T, C) and `group_id` is (B, N). Only the camera and register
    tokens are read, so the patch tokens are dropped before the per-group copies.
    """
    if group_id.ndim != 2:
        raise ValueError(f"Expected group ids (B, N), got {tuple(group_id.shape)}")
    if tokens.shape[:2] != group_id.shape:
        raise ValueError(f"Tokens {tuple(tokens.shape[:2])} do not match group ids {tuple(group_id.shape)}")

    tokens = tokens[:, :, :patch_token_start]
    rows = []
    for batch_index in range(group_id.shape[0]):
        pose = None
        for group in torch.unique(group_id[batch_index], sorted=True):
            index = torch.nonzero(group_id[batch_index] == group, as_tuple=False).flatten()
            part = tokens[batch_index].index_select(0, index).unsqueeze(0)
            predicted = camera_head([part], patch_token_start)
            if pose is None:
                pose = predicted.new_zeros(group_id.shape[1], predicted.shape[-1])
            pose = pose.index_copy(0, index, predicted[0])
        rows.append(pose)
    return torch.stack(rows, dim=0)


@torch.no_grad()
def scene_groups(affinity: torch.Tensor, threshold: float, linkage: str = "average") -> torch.Tensor:
    """Agglomerative clustering of an affinity matrix (B, N, N) into group ids (B, N).

    Clusters merge while their linkage affinity is at least `threshold`. Average linkage
    needs most pairs of two clusters to agree, so a single false positive pair does not
    merge two scenes. Ids start at 0 and follow the first frame of each group.
    """
    if linkage not in LINKAGES:
        raise ValueError(f"linkage must be one of {LINKAGES}, got {linkage!r}")
    rows = [_cluster(row.float().cpu(), threshold, linkage) for row in affinity]
    return torch.stack(rows, dim=0).to(affinity.device)


def _cluster(affinity: torch.Tensor, threshold: float, linkage: str) -> torch.Tensor:
    affinity = 0.5 * (affinity + affinity.T)
    num_frames = affinity.shape[0]
    link = affinity.clone()
    size = [1] * num_frames
    members = [[frame] for frame in range(num_frames)]
    active = list(range(num_frames))
    while len(active) > 1:
        block = link[active][:, active].clone()
        block.fill_diagonal_(float("-inf"))
        first, second = divmod(int(block.argmax()), len(active))
        if block[first, second] < threshold:
            break
        keep, drop = active[first], active[second]
        if linkage == "average":
            merged = (link[keep] * size[keep] + link[drop] * size[drop]) / (size[keep] + size[drop])
        elif linkage == "single":
            merged = torch.maximum(link[keep], link[drop])
        else:
            merged = torch.minimum(link[keep], link[drop])
        link[keep, :] = merged
        link[:, keep] = merged
        link[keep, keep] = 1.0
        size[keep] += size[drop]
        members[keep] += members[drop]
        active.remove(drop)

    labels = [0] * num_frames
    for cluster in active:
        for frame in members[cluster]:
            labels[frame] = cluster
    order: dict[int, int] = {}
    return torch.tensor([order.setdefault(label, len(order)) for label in labels], dtype=torch.long)


def _device_type(tensor: torch.Tensor) -> str:
    return tensor.device.type if tensor.device.type in ("cuda", "mps", "cpu") else "cpu"
