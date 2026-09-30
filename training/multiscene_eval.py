"""Score a mixed bag: clustering, per-scene pose, per-scene depth.

Three forwards of the same model:

- plain: the camera head sees every frame, as the original model does
- predicted: the camera head is isolated with the affinity groups
- oracle: each ground-truth scene is forwarded on its own

Contamination is how much worse `plain` is than `oracle` on the same images.
"""

import torch

from losses.metric import rotation_angle, translation_angle
from vggt_omega.models.multiscene import scene_groups
from vggt_omega.utils.geometry import closed_form_inverse_se3
from vggt_omega.utils.pose_enc import encoding_to_camera


def clustering_scores(affinity: torch.Tensor, scene_id: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Off-diagonal pair F1, plus agreement of the connected components."""
    affinity, scene_id = _as_single(affinity, scene_id)
    num_frames = scene_id.shape[0]
    eye = torch.eye(num_frames, device=affinity.device, dtype=torch.bool)
    predicted_same = (affinity >= threshold)[~eye]
    target_same = (scene_id[:, None] == scene_id[None, :])[~eye]
    true_positive = (predicted_same & target_same).sum().float()
    precision = true_positive / predicted_same.sum().clamp(min=1).float()
    recall = true_positive / target_same.sum().clamp(min=1).float()
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
    groups = scene_groups(affinity.unsqueeze(0), threshold)[0]
    group_same = (groups[:, None] == groups[None, :])[~eye]
    agreement = (group_same == target_same).float().mean()
    return {
        "affinity_f1": float(f1),
        "group_agreement": float(agreement),
    }


def pose_scores(predicted_extrinsics: torch.Tensor, target_extrinsics: torch.Tensor, scene_id: torch.Tensor) -> dict[str, float]:
    """Mean pairwise rotation and translation error, in degrees, averaged over scenes."""
    predicted_extrinsics, target_extrinsics, scene_id = _pose_single(
        predicted_extrinsics, target_extrinsics, scene_id
    )
    rotations = []
    translations = []
    for scene in torch.unique(scene_id, sorted=True):
        index = torch.nonzero(scene_id == scene, as_tuple=False).flatten()
        if index.numel() < 2:
            continue
        rotation, translation = _pairwise_pose_error(
            predicted_extrinsics.index_select(0, index),
            target_extrinsics.index_select(0, index),
        )
        rotations.append(rotation.mean())
        translations.append(translation.mean())
    if not rotations:
        raise ValueError("Pose scoring needs at least one scene with two frames")
    return {
        "rotation_deg": float(torch.stack(rotations).mean()),
        "translation_deg": float(torch.stack(translations).mean()),
    }


def depth_scores(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scene_id: torch.Tensor,
    min_valid: int = 10,
) -> dict[str, float]:
    """Per-scene median-scale absolute relative error and the 1.25 depth threshold."""
    predicted = _depth_maps(predicted)
    target = _depth_maps(target)
    if predicted.shape != target.shape:
        raise ValueError(f"Depth shapes differ: {tuple(predicted.shape)} vs {tuple(target.shape)}")
    scene_id = scene_id.view(-1)
    abs_rel = []
    delta = []
    for scene in torch.unique(scene_id, sorted=True):
        index = torch.nonzero(scene_id == scene, as_tuple=False).flatten()
        valid = torch.isfinite(target[index]) & (target[index] > 1e-6) & torch.isfinite(predicted[index])
        valid = valid & (predicted[index] > 1e-6)
        if int(valid.sum()) < min_valid:
            continue
        pred_values = predicted[index][valid]
        target_values = target[index][valid]
        scale = (target_values / pred_values).median()
        aligned = pred_values * scale
        abs_rel.append((aligned - target_values).abs().div(target_values).mean())
        ratio = torch.maximum(aligned / target_values, target_values / aligned)
        delta.append((ratio < 1.25).float().mean())
    if not abs_rel:
        raise ValueError("Depth scoring found no scene with enough valid pixels")
    return {
        "abs_rel": float(torch.stack(abs_rel).mean()),
        "delta125": float(torch.stack(delta).mean()),
    }


def contamination(plain: dict, oracle: dict) -> dict[str, float]:
    """How much the mixed bag hurts each shared metric. Larger means more mixing."""
    shared = set(plain) & set(oracle)
    return {key: float(plain[key]) - float(oracle[key]) for key in sorted(shared)}


@torch.inference_mode()
def run_systems(
    model: torch.nn.Module,
    images: torch.Tensor,
    scene_id: torch.Tensor,
    target_extrinsics: torch.Tensor,
    target_depth: torch.Tensor,
    threshold: float = 0.5,
    min_valid: int = 10,
) -> dict:
    """Score plain, predicted-group, and oracle forwards of one bag.

    `images` is (N, 3, H, W) or (1, N, 3, H, W). `scene_id` is (N,).
    """
    images = _batch_images(images)
    scene_id = scene_id.view(-1).to(device=images.device)
    image_hw = tuple(images.shape[-2:])

    plain = _score_forward(
        _forward(model, images, isolate=False),
        scene_id,
        target_extrinsics,
        target_depth,
        image_hw,
        threshold,
        min_valid,
    )
    predicted = _score_forward(
        _forward(model, images, isolate=True),
        scene_id,
        target_extrinsics,
        target_depth,
        image_hw,
        threshold,
        min_valid,
    )
    oracle = _score_forward(
        _oracle_forward(model, images, scene_id),
        scene_id,
        target_extrinsics,
        target_depth,
        image_hw,
        threshold,
        min_valid,
    )
    return {
        "plain": plain,
        "predicted": predicted,
        "oracle": oracle,
        "contamination": contamination(plain, oracle),
    }


def _score_forward(prediction, scene_id, target_extrinsics, target_depth, image_hw, threshold, min_valid):
    extrinsics, _ = encoding_to_camera(prediction["pose_enc"], image_hw)
    scores = pose_scores(extrinsics[0], target_extrinsics, scene_id)
    scores.update(depth_scores(prediction["depth"], target_depth, scene_id, min_valid=min_valid))
    if "affinity" in prediction:
        scores.update(clustering_scores(prediction["affinity"], scene_id, threshold))
    return scores


def _forward(model, images, isolate: bool):
    previous = getattr(model, "isolate_camera", None)
    if previous is not None:
        model.isolate_camera = isolate
    try:
        return model(images)
    finally:
        if previous is not None:
            model.isolate_camera = previous


def _oracle_forward(model, images, scene_id):
    num_frames = images.shape[1]
    pose = None
    depth = None
    for scene in torch.unique(scene_id, sorted=True):
        index = torch.nonzero(scene_id == scene, as_tuple=False).flatten()
        prediction = model(images.index_select(1, index))
        if pose is None:
            pose = prediction["pose_enc"].new_zeros(1, num_frames, prediction["pose_enc"].shape[-1])
            depth_shape = _depth_maps(prediction["depth"]).shape
            depth = prediction["depth"].new_zeros(1, num_frames, *depth_shape[1:])
        pose = pose.index_copy(1, index, prediction["pose_enc"])
        depth = depth.index_copy(1, index, _depth_maps(prediction["depth"]).unsqueeze(0))
    return {"pose_enc": pose, "depth": depth}


def _pairwise_pose_error(predicted: torch.Tensor, target: torch.Tensor):
    first, second = torch.triu_indices(predicted.shape[0], predicted.shape[0], offset=1)
    predicted_relative = _relative(predicted, first, second)
    target_relative = _relative(target, first, second)
    rotation = rotation_angle(target_relative[:, :3, :3], predicted_relative[:, :3, :3])
    translation = translation_angle(target_relative[:, :3, 3], predicted_relative[:, :3, 3])
    # A pair with no translation has no direction. Angular error is meaningless there.
    predicted_translation = predicted_relative[:, :3, 3]
    target_translation = target_relative[:, :3, 3]
    both_still = (predicted_translation.norm(dim=-1) < 1e-6) & (target_translation.norm(dim=-1) < 1e-6)
    translation = torch.where(both_still, torch.zeros_like(translation), translation)
    return rotation, translation


def _relative(extrinsics: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    inverse = closed_form_inverse_se3(extrinsics.index_select(0, first))
    return _as_4x4(extrinsics.index_select(0, second)).matmul(inverse)


def _as_4x4(extrinsics: torch.Tensor) -> torch.Tensor:
    poses = extrinsics.new_zeros(extrinsics.shape[0], 4, 4)
    poses[:, :3] = extrinsics
    poses[:, 3, 3] = 1
    return poses


def _batch_images(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        return images.unsqueeze(0)
    if images.ndim != 5:
        raise ValueError(f"Expected images (N, 3, H, W) or (1, N, 3, H, W), got {tuple(images.shape)}")
    return images


def _as_single(affinity, scene_id):
    if affinity.ndim == 3:
        affinity = affinity[0]
    return affinity, scene_id.view(-1)


def _pose_single(predicted, target, scene_id):
    if predicted.ndim == 4:
        predicted = predicted[0]
    if target.ndim == 4:
        target = target[0]
    return predicted, target, scene_id.view(-1)


def _depth_maps(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim == 3:
        return depth
    if depth.ndim == 4 and depth.shape[0] == 1:
        return depth[0]
    raise ValueError(f"Expected depth (N, H, W), got {tuple(depth.shape)}")
