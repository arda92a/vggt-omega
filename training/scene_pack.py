"""Turn one mixed forward pass into the per-scene output from the plan.

Each scene gets the frame indices, an anchor frame, camera poses relative to
that anchor, and a point cloud in the anchor camera frame.
"""

import torch

from losses.unprojection import unproject_depth_to_points_torch_batch
from vggt_omega.utils.geometry import closed_form_inverse_se3
from vggt_omega.utils.pose_enc import encoding_to_camera


def pack_scenes(affinity, group_id, pose_enc, depth, image_hw) -> list[dict]:
    """Pack one bag.

    affinity: (N, N), group_id: (N,), pose_enc: (N, 9), depth: (N, H, W) or (N, H, W, 1).
    """
    affinity, group_id, pose_enc, depth = _single_bag(affinity, group_id, pose_enc, depth)
    extrinsics, _ = encoding_to_camera(pose_enc.unsqueeze(0), image_hw)
    extrinsics = extrinsics[0]
    scenes = []
    for scene in torch.unique(group_id, sorted=True):
        index = torch.nonzero(group_id == scene, as_tuple=False).flatten()
        local_anchor = int(anchor_offset(affinity, index))
        anchor = int(index[local_anchor])
        poses = poses_relative_to_anchor(extrinsics.index_select(0, index), local_anchor)
        points = points_in_anchor_frame(
            depth.index_select(0, index),
            extrinsics.index_select(0, index),
            encoding_to_camera(pose_enc.index_select(0, index).unsqueeze(0), image_hw)[1][0],
            local_anchor,
        )
        scenes.append(
            {
                "images": [int(frame) for frame in index],
                "anchor": anchor,
                "camera_poses": poses,
                "pointcloud": points,
            }
        )
    return scenes


def anchor_offset(affinity: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Index inside `index` of the frame with the highest mean affinity.

    Ties keep the earliest frame. `index` is the group's frame positions in order.
    """
    block = affinity.index_select(0, index).index_select(1, index)
    scores = block.mean(dim=1)
    return torch.argmax(scores)


def poses_relative_to_anchor(extrinsics: torch.Tensor, anchor: int) -> torch.Tensor:
    """Camera-from-anchor extrinsics. The anchor row is identity."""
    inverse_anchor = closed_form_inverse_se3(extrinsics[anchor : anchor + 1])
    relative = _as_4x4(extrinsics).matmul(inverse_anchor)
    return relative[:, :3]


def points_in_anchor_frame(depth, extrinsics, intrinsics, anchor: int) -> torch.Tensor:
    """Unproject depth and express the points in the anchor camera."""
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


def _single_bag(affinity, group_id, pose_enc, depth):
    if affinity.ndim == 3:
        affinity = affinity[0]
    if group_id.ndim == 2:
        group_id = group_id[0]
    if pose_enc.ndim == 3:
        pose_enc = pose_enc[0]
    if depth.ndim >= 4 and depth.shape[0] == 1 and affinity.shape[0] != 1:
        depth = depth[0]
    if depth.ndim == 5:
        depth = depth[0]
    return affinity, group_id, pose_enc, depth


def _as_4x4(extrinsics: torch.Tensor) -> torch.Tensor:
    poses = extrinsics.new_zeros(extrinsics.shape[0], 4, 4)
    poses[:, :3] = extrinsics
    poses[:, 3, 3] = 1
    return poses
