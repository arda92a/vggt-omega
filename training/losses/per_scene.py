"""Evaluate a frame loss separately on each scene, then average.

A mixed bag stores several scenes in one tensor. Camera pairs and depth reductions must
not cross a scene boundary. Ground-truth scene ids define the slices. Predicted groups
are not used here: a wrong group would supervise a camera against another scene's
geometry.

The network predicts each scene's cameras in its own gauge: nothing tells it which frame
of a scene is the origin. `anchor_to_first_frame` therefore expresses prediction and
ground truth relative to the same frame of the slice before any pose or point loss.
"""

import torch

from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.rotation import mat_to_quat
from vggt_omega.utils.scenes import poses_relative_to_anchor


_FRAME_KEYS = (
    "images",
    "depths",
    "extrinsics",
    "intrinsics",
    "world_points",
    "point_masks",
    "tracks",
    "track_vis_mask",
)
_ROW_KEYS = ("valid_seq_mask", "depth_train_mask", "is_synthetic")
_PREDICTION_KEYS = ("depth", "depth_conf", "world_points")


def mean_over_scenes(loss_fn, predictions, batch, scene_id: torch.Tensor) -> dict:
    """Run `loss_fn(predictions, batch)` on every scene slice and average the dict."""
    if scene_id.ndim != 2:
        raise ValueError(f"Expected scene_id (B, N), got {tuple(scene_id.shape)}")

    scene_losses = []
    for batch_index in range(scene_id.shape[0]):
        for scene in torch.unique(scene_id[batch_index], sorted=True):
            frame_index = torch.nonzero(scene_id[batch_index] == scene, as_tuple=False).flatten()
            scene_predictions, scene_batch = _slice_scene(predictions, batch, batch_index, frame_index)
            scene_losses.append(loss_fn(scene_predictions, scene_batch))
    if not scene_losses:
        raise ValueError("scene_id did not contain any scene")
    return _average_dicts(scene_losses)


def anchor_to_first_frame(predictions: dict, batch: dict) -> tuple[dict, dict]:
    """Re-express predicted and ground-truth cameras of one scene slice relative to its first frame.

    Returns copies. The first frame becomes identity in both, so it carries no pose loss,
    and every other frame is compared in a gauge both sides share.
    """
    pose = predictions["pose_enc_list"][-1]
    if pose.shape[0] != 1:
        raise ValueError(f"Expected one bag per scene slice, got batch {pose.shape[0]}")
    image_hw = batch["images"].shape[-2:]

    predicted, _ = encoding_to_camera(pose, image_hw, build_intrinsics=False)
    relative = poses_relative_to_anchor(predicted[0], 0)
    anchored = torch.cat([relative[:, :, 3], mat_to_quat(relative[:, :, :3]), pose[0, :, 7:]], dim=-1)

    target = poses_relative_to_anchor(batch["extrinsics"][0], 0).unsqueeze(0)
    return (
        {**predictions, "pose_enc_list": [anchored.unsqueeze(0)]},
        {**batch, "extrinsics": target},
    )


def _slice_scene(predictions, batch, batch_index: int, frame_index: torch.Tensor):
    num_frames = batch["scene_id"].shape[1]
    sliced_predictions = {}
    if "pose_enc_list" in predictions:
        sliced_predictions["pose_enc_list"] = [
            _index_frames(pose, batch_index, frame_index, num_frames)
            for pose in predictions["pose_enc_list"]
        ]
    for key in _PREDICTION_KEYS:
        if key in predictions and torch.is_tensor(predictions[key]):
            sliced_predictions[key] = _index_frames(predictions[key], batch_index, frame_index, num_frames)

    sliced_batch = {}
    for key in _FRAME_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == num_frames:
            sliced_batch[key] = _index_frames(value, batch_index, frame_index, num_frames)
    for key in _ROW_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[0] == batch["scene_id"].shape[0]:
            sliced_batch[key] = value[batch_index : batch_index + 1]
    return sliced_predictions, sliced_batch


def _index_frames(value: torch.Tensor, batch_index: int, frame_index: torch.Tensor, num_frames: int) -> torch.Tensor:
    if value.shape[0] <= batch_index or value.ndim < 2 or value.shape[1] != num_frames:
        raise ValueError(
            f"Cannot slice frames from shape {tuple(value.shape)} at batch {batch_index}, N={num_frames}"
        )
    frame_index = frame_index.to(device=value.device)
    return value[batch_index : batch_index + 1].index_select(1, frame_index)


def _average_dicts(losses: list[dict]) -> dict:
    averaged = {}
    for key in losses[0]:
        values = [item[key] for item in losses]
        if torch.is_tensor(values[0]):
            scalars = []
            for value in values:
                if value.numel() != 1:
                    raise ValueError(f"Per-scene value '{key}' must be a scalar, got shape {tuple(value.shape)}")
                scalars.append(value.reshape(()))
            averaged[key] = torch.stack(scalars).mean()
        else:
            averaged[key] = sum(float(value) for value in values) / len(values)
    return averaged
