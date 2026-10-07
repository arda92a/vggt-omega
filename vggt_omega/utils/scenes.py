"""Per-scene output of a mixed forward pass, and clustering metrics for scene groups.

Each scene gets its frame indices, an anchor frame, camera poses relative to that anchor,
and a point cloud in the anchor camera frame.
"""

import torch

from vggt_omega.utils.geometry import closed_form_inverse_se3, unproject_depth_to_points_torch_batch
from vggt_omega.utils.pose_enc import encoding_to_camera


def pack_scenes(
    affinity: torch.Tensor,
    group_id: torch.Tensor,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    image_hw: tuple[int, int],
) -> list[dict]:
    """Pack one bag. `affinity` (N, N), `group_id` (N,), `pose_enc` (N, 9), `depth` (N, H, W[, 1])."""
    extrinsics, intrinsics = encoding_to_camera(pose_enc.unsqueeze(0), image_hw)
    extrinsics, intrinsics = extrinsics[0], intrinsics[0]
    scenes = []
    for scene in torch.unique(group_id, sorted=True):
        index = torch.nonzero(group_id == scene, as_tuple=False).flatten()
        local_anchor = int(anchor_offset(affinity, index))
        scene_extrinsics = extrinsics.index_select(0, index)
        scenes.append(
            {
                "images": [int(frame) for frame in index],
                "anchor": int(index[local_anchor]),
                "camera_poses": poses_relative_to_anchor(scene_extrinsics, local_anchor),
                "pointcloud": points_in_anchor_frame(
                    depth.index_select(0, index),
                    scene_extrinsics,
                    intrinsics.index_select(0, index),
                    local_anchor,
                ),
            }
        )
    return scenes


def anchor_offset(affinity: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Position inside `index` of the frame with the highest mean affinity. Ties keep the earliest."""
    block = affinity.index_select(0, index).index_select(1, index)
    return torch.argmax(block.mean(dim=1))


def poses_relative_to_anchor(extrinsics: torch.Tensor, anchor: int) -> torch.Tensor:
    """Camera-from-anchor extrinsics (N, 3, 4). The anchor row is identity."""
    inverse_anchor = closed_form_inverse_se3(extrinsics[anchor : anchor + 1])
    return as_4x4(extrinsics).matmul(inverse_anchor)[:, :3]


def points_in_anchor_frame(
    depth: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    anchor: int,
) -> torch.Tensor:
    """Unproject depth (N, H, W[, 1]) and express the points in the anchor camera."""
    if depth.ndim == 4:
        depth = depth.squeeze(-1)
    world = unproject_depth_to_points_torch_batch(
        depth.unsqueeze(0),
        extrinsics.unsqueeze(0),
        intrinsics.unsqueeze(0),
    )[0]
    rotation = extrinsics[anchor, :3, :3]
    translation = extrinsics[anchor, :3, 3]
    return torch.einsum("khwc,dc->khwd", world, rotation) + translation


def as_4x4(extrinsics: torch.Tensor) -> torch.Tensor:
    poses = extrinsics.new_zeros(*extrinsics.shape[:-2], 4, 4)
    poses[..., :3, :] = extrinsics
    poses[..., 3, 3] = 1
    return poses


def adjusted_rand_index(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Adjusted Rand index of two label vectors (N,). 1 is identical partitions, 0 is chance."""
    if predicted.shape != target.shape or predicted.ndim != 1:
        raise ValueError(f"Expected two label vectors of equal length, got {tuple(predicted.shape)}, {tuple(target.shape)}")
    num_items = predicted.numel()
    if num_items < 2:
        return 1.0
    _, predicted = torch.unique(predicted, return_inverse=True)
    _, target = torch.unique(target, return_inverse=True)
    table = torch.zeros(int(predicted.max()) + 1, int(target.max()) + 1, dtype=torch.float64)
    table.index_put_((predicted, target), torch.ones(num_items, dtype=torch.float64), accumulate=True)

    def pairs(count: torch.Tensor) -> torch.Tensor:
        return (count * (count - 1) / 2).sum()

    index = pairs(table)
    rows = pairs(table.sum(dim=1))
    columns = pairs(table.sum(dim=0))
    total = num_items * (num_items - 1) / 2
    expected = rows * columns / total
    maximum = 0.5 * (rows + columns)
    if float(maximum - expected) == 0.0:
        return 1.0
    return float((index - expected) / (maximum - expected))
