"""Score a mixed bag: scene clustering, per-scene pose, per-scene depth.

Systems compared on the same images:

- predicted: routing and camera isolation with the predicted groups. This is inference.
- oracle:    the same model with ground-truth groups. The gap to `predicted` is the cost of
             clustering errors.
- plain:     no isolation, every frame in one group. The gap to `oracle` is contamination.
- separate:  each ground-truth scene forwarded alone, which is the original model's behaviour.

Pose is scored on pairs inside a ground-truth scene, so it is independent of the gauge each
scene is predicted in. A pair split across two predicted groups is penalised, as it should be.
"""

import torch

from losses.metric import rotation_angle, translation_angle
from vggt_omega.utils.geometry import closed_form_inverse_se3
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.scenes import adjusted_rand_index, as_4x4

POSE_KEYS = ("rotation_deg", "translation_deg", "rra_5", "rra_15", "rta_5", "rta_15", "auc_30")
DEPTH_KEYS = ("abs_rel", "delta125", "abs_rel_raw")


def clustering_scores(prediction: dict, scene_id: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Pair F1, pair agreement, ARI, exact partition and scene-count error, for final and route groups."""
    scene_id = scene_id.view(-1).cpu()
    num_frames = scene_id.shape[0]
    off = ~torch.eye(num_frames, dtype=torch.bool)
    target_same = (scene_id[:, None] == scene_id[None, :])[off]
    scores = {}
    for prefix, affinity_key, group_key in (
        ("", "affinity", "group_id"),
        ("route_", "route_affinity", "route_group_id"),
    ):
        if affinity_key not in prediction:
            continue
        affinity = prediction[affinity_key].reshape(num_frames, num_frames).float().cpu()
        groups = prediction[group_key].reshape(num_frames).cpu()
        predicted_same = (affinity >= threshold)[off]
        true_positive = (predicted_same & target_same).sum().float()
        precision = true_positive / predicted_same.sum().clamp(min=1).float()
        recall = true_positive / target_same.sum().clamp(min=1).float()
        group_same = (groups[:, None] == groups[None, :])[off]
        scores[f"{prefix}affinity_f1"] = float(2 * precision * recall / (precision + recall).clamp(min=1e-6))
        scores[f"{prefix}group_agreement"] = float((group_same == target_same).float().mean())
        scores[f"{prefix}ari"] = adjusted_rand_index(groups, scene_id)
        scores[f"{prefix}exact_partition"] = float(bool((group_same == target_same).all()))
        scores[f"{prefix}scene_count_error"] = float(abs(groups.unique().numel() - scene_id.unique().numel()))
    return scores


def pose_scores(predicted: torch.Tensor, target: torch.Tensor, scene_id: torch.Tensor) -> dict[str, float]:
    """Pairwise relative pose errors inside each scene: mean degrees, accuracy at 5/15 degrees, AUC@30."""
    predicted, target = _extrinsics(predicted), _extrinsics(target)
    scene_id = scene_id.view(-1).to(predicted.device)
    rotations, translations = [], []
    for scene in torch.unique(scene_id, sorted=True):
        index = torch.nonzero(scene_id == scene, as_tuple=False).flatten()
        if index.numel() < 2:
            continue
        rotation, translation = _pairwise_pose_error(
            predicted.index_select(0, index), target.index_select(0, index.to(target.device))
        )
        rotations.append(rotation)
        translations.append(translation)
    if not rotations:
        raise ValueError("Pose scoring needs at least one scene with two frames")
    rotation, translation = torch.cat(rotations), torch.cat(translations)
    worst = torch.maximum(rotation, translation)
    thresholds = torch.arange(1, 31, dtype=worst.dtype, device=worst.device)
    return {
        "rotation_deg": float(rotation.mean()),
        "translation_deg": float(translation.mean()),
        "rra_5": float((rotation < 5).float().mean()),
        "rra_15": float((rotation < 15).float().mean()),
        "rta_5": float((translation < 5).float().mean()),
        "rta_15": float((translation < 15).float().mean()),
        "auc_30": float((worst[None] <= thresholds[:, None]).float().mean()),
    }


def depth_scores(predicted: torch.Tensor, target: torch.Tensor, scene_id: torch.Tensor, min_valid: int = 10) -> dict[str, float]:
    """Per-scene median-scaled abs-rel and delta<1.25, plus abs-rel with no scale alignment."""
    predicted, target = _depth_maps(predicted), _depth_maps(target)
    if predicted.shape != target.shape:
        raise ValueError(f"Depth shapes differ: {tuple(predicted.shape)} vs {tuple(target.shape)}")
    scene_id = scene_id.view(-1).to(predicted.device)
    abs_rel, delta, abs_rel_raw = [], [], []
    for scene in torch.unique(scene_id, sorted=True):
        index = torch.nonzero(scene_id == scene, as_tuple=False).flatten()
        pred, gt = predicted[index], target[index]
        valid = torch.isfinite(gt) & (gt > 1e-6) & torch.isfinite(pred) & (pred > 1e-6)
        if int(valid.sum()) < min_valid:
            continue
        pred, gt = pred[valid], gt[valid]
        aligned = pred * (gt / pred).median()
        abs_rel.append((aligned - gt).abs().div(gt).mean())
        delta.append((torch.maximum(aligned / gt, gt / aligned) < 1.25).float().mean())
        abs_rel_raw.append((pred - gt).abs().div(gt).mean())
    if not abs_rel:
        raise ValueError("Depth scoring found no scene with enough valid pixels")
    return {
        "abs_rel": float(torch.stack(abs_rel).mean()),
        "delta125": float(torch.stack(delta).mean()),
        "abs_rel_raw": float(torch.stack(abs_rel_raw).mean()),
    }


def score_prediction(prediction, scene_id, target_extrinsics, target_depth, image_hw, threshold=0.5, min_valid=10) -> dict:
    extrinsics, _ = encoding_to_camera(prediction["pose_enc"], image_hw, build_intrinsics=False)
    scores = pose_scores(extrinsics[0], target_extrinsics, scene_id)
    scores.update(depth_scores(prediction["depth"], target_depth, scene_id, min_valid=min_valid))
    if "affinity" in prediction:
        scores.update(clustering_scores(prediction, scene_id, threshold))
    return scores


def difference(first: dict, second: dict, keys=POSE_KEYS + DEPTH_KEYS) -> dict[str, float]:
    """first - second on the metrics both report."""
    return {key: float(first[key]) - float(second[key]) for key in keys if key in first and key in second}


@torch.inference_mode()
def run_systems(
    model: torch.nn.Module,
    images: torch.Tensor,
    scene_id: torch.Tensor,
    target_extrinsics: torch.Tensor,
    target_depth: torch.Tensor,
    threshold: float = 0.5,
    min_valid: int = 10,
    include_separate: bool = True,
) -> dict:
    """Score the predicted, oracle, plain (and optionally separate) forwards of one bag.

    `images` is (N, 3, H, W) or (1, N, 3, H, W). `scene_id` is (N,).
    """
    images = _batch_images(images)
    scene_id = scene_id.view(-1).to(device=images.device)
    labels = scene_id.unsqueeze(0)
    image_hw = tuple(images.shape[-2:])

    def score(prediction):
        return score_prediction(prediction, scene_id, target_extrinsics, target_depth, image_hw, threshold, min_valid)

    systems = {
        "predicted": score(model(images)),
        "oracle": score(model(images, scene_id=labels, use_gt_groups=True)),
        "plain": score(model(images, isolate=False)),
    }
    if include_separate:
        systems["separate"] = score(_separate_forward(model, images, scene_id))
    report = dict(systems)
    report["contamination"] = difference(systems["plain"], systems["oracle"])
    report["cluster_gap"] = difference(systems["predicted"], systems["oracle"])
    return report


def _separate_forward(model, images, scene_id):
    num_frames = images.shape[1]
    pose = depth = None
    for scene in torch.unique(scene_id, sorted=True):
        index = torch.nonzero(scene_id == scene, as_tuple=False).flatten()
        prediction = model(images.index_select(1, index), isolate=False)
        if pose is None:
            pose = prediction["pose_enc"].new_zeros(1, num_frames, prediction["pose_enc"].shape[-1])
            depth = prediction["depth"].new_zeros(1, num_frames, *_depth_maps(prediction["depth"]).shape[1:])
        pose = pose.index_copy(1, index, prediction["pose_enc"])
        depth = depth.index_copy(1, index, _depth_maps(prediction["depth"]).unsqueeze(0))
    return {"pose_enc": pose, "depth": depth}


def _pairwise_pose_error(predicted: torch.Tensor, target: torch.Tensor):
    first, second = torch.triu_indices(predicted.shape[0], predicted.shape[0], offset=1, device=predicted.device)
    predicted_relative = _relative(predicted, first, second)
    target_relative = _relative(target, first.to(target.device), second.to(target.device))
    rotation = rotation_angle(target_relative[:, :3, :3], predicted_relative[:, :3, :3])
    translation = translation_angle(target_relative[:, :3, 3], predicted_relative[:, :3, 3])
    # A pair with no translation has no direction, so its angular error is meaningless.
    both_still = (predicted_relative[:, :3, 3].norm(dim=-1) < 1e-6) & (target_relative[:, :3, 3].norm(dim=-1) < 1e-6)
    return rotation, torch.where(both_still, torch.zeros_like(translation), translation)


def _relative(extrinsics: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    inverse = closed_form_inverse_se3(extrinsics.index_select(0, first))
    return as_4x4(extrinsics.index_select(0, second)).matmul(inverse)


def _extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    return extrinsics[0] if extrinsics.ndim == 4 else extrinsics


def _batch_images(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        return images.unsqueeze(0)
    if images.ndim != 5:
        raise ValueError(f"Expected images (N, 3, H, W) or (1, N, 3, H, W), got {tuple(images.shape)}")
    return images


def _depth_maps(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3:
        return depth
    raise ValueError(f"Expected depth (N, H, W), got {tuple(depth.shape)}")
