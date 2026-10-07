# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def closed_form_inverse_se3(se3, R=None, T=None):
    """Invert a batch of 3x4 or 4x4 SE(3) matrices."""
    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must have shape (N, 4, 4) or (N, 3, 4), got {se3.shape}")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_t = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_t, T)
        inverted = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_t = R.transpose(1, 2)
        top_right = -torch.bmm(R_t, T)
        inverted = torch.eye(4, device=R.device, dtype=R.dtype)[None].repeat(len(R), 1, 1)

    inverted[:, :3, :3] = R_t
    inverted[:, :3, 3:] = top_right
    return inverted


def unproject_depth_to_cam_coords(depth, intrinsics):
    """Unproject COLMAP-convention depth maps into camera coordinates."""
    if depth.ndim == 4:
        depth = depth.unsqueeze(-1)
    if depth.ndim != 5:
        raise ValueError("depth must have shape (B, S, H, W[, 1])")

    batch_size, num_frames, height, width, _ = depth.shape
    intrinsics = intrinsics.to(device=depth.device, dtype=depth.dtype)
    u = (
        torch.arange(width, device=depth.device, dtype=depth.dtype).view(1, 1, 1, width)
        + 0.5
    )
    v = (
        torch.arange(height, device=depth.device, dtype=depth.dtype).view(
            1, 1, height, 1
        )
        + 0.5
    )

    focal_x = intrinsics[..., 0, 0].view(batch_size, num_frames, 1, 1)
    focal_y = intrinsics[..., 1, 1].view(batch_size, num_frames, 1, 1)
    center_x = intrinsics[..., 0, 2].view(batch_size, num_frames, 1, 1)
    center_y = intrinsics[..., 1, 2].view(batch_size, num_frames, 1, 1)
    eps = torch.finfo(depth.dtype).eps
    focal_x = focal_x.clamp(min=eps)
    focal_y = focal_y.clamp(min=eps)

    z = depth.squeeze(-1)
    x = (u - center_x) * z / focal_x
    y = (v - center_y) * z / focal_y
    return torch.stack((x, y, z), dim=-1)


def unproject_depth_to_points_torch_batch(depth, extrinsics, intrinsics):
    """Unproject depth maps into world coordinates."""
    camera_points = unproject_depth_to_cam_coords(depth, intrinsics)
    batch_size, num_frames = camera_points.shape[:2]
    extrinsics = extrinsics.to(device=camera_points.device, dtype=camera_points.dtype)
    camera_to_world = closed_form_inverse_se3(extrinsics.reshape(-1, 3, 4))
    rotation = camera_to_world[:, :3, :3].reshape(batch_size, num_frames, 3, 3)
    translation = camera_to_world[:, :3, 3].reshape(batch_size, num_frames, 3)
    return (
        torch.einsum("bshwc,bscd->bshwd", camera_points, rotation.transpose(-1, -2))
        + translation[:, :, None, None, :]
    )
