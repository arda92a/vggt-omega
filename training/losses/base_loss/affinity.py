"""Affinity loss and clustering metrics.

The loss is a class-balanced BCE over off-diagonal pairs. With S scenes only 1/S of the
pairs are positive, and the diagonal is always 1, so a plain mean rewards "everything is
the same scene". The route head (middle layer) and the final head are supervised
separately, and optionally the final head again after scene attention.
"""

import torch
import torch.nn.functional as F

from vggt_omega.utils.scenes import adjusted_rand_index


def compute_affinity_loss(
    predictions,
    batch,
    include_output: bool = False,
    route_weight: float = 1.0,
) -> dict:
    """Balanced BCE for the final affinity, the route affinity, and optionally the output affinity."""
    for key in ("affinity_logits", "route_affinity_logits"):
        if key not in predictions:
            raise KeyError(f"Affinity loss requires predictions['{key}']. Enable model.multiscene.")
    if "affinity" not in batch:
        raise KeyError(
            "Affinity loss requires batch['affinity']. Enable data.train.mixed_scene with num_scenes >= 2."
        )

    target = batch["affinity"].to(dtype=predictions["affinity_logits"].dtype)
    final_loss = balanced_bce(predictions["affinity_logits"], target)
    route_loss = balanced_bce(predictions["route_affinity_logits"], target)
    total = final_loss + route_weight * route_loss
    logs = {
        "loss_affinity_final": final_loss.detach(),
        "loss_affinity_route": route_loss.detach(),
        "affinity_f1": _off_diagonal_f1(predictions["affinity"], target),
        "route_f1": _off_diagonal_f1(predictions["route_affinity"], target),
    }
    if include_output:
        if "affinity_output_logits" not in predictions:
            raise KeyError("include_output requires predictions['affinity_output_logits']")
        output_loss = balanced_bce(predictions["affinity_output_logits"], target)
        total = total + output_loss
        logs["loss_affinity_output"] = output_loss.detach()
        logs["affinity_output_f1"] = _off_diagonal_f1(predictions["affinity_output"], target)
    logs["loss_affinity"] = total

    if "scene_id" in batch:
        for name, key in (("", "group_id"), ("route_", "route_group_id")):
            if key in predictions:
                logs.update(_partition_scores(predictions[key], batch["scene_id"], prefix=name))
    return logs


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of the positive-pair BCE and the negative-pair BCE over off-diagonal pairs."""
    off = ~torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool)
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")[:, off]
    positive = target[:, off] >= 0.5
    terms = [loss[mask].mean() for mask in (positive, ~positive) if mask.any()]
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _off_diagonal_f1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    off = ~torch.eye(prediction.shape[-1], device=prediction.device, dtype=torch.bool)
    predicted_same = (prediction >= 0.5)[:, off]
    target_same = (target >= 0.5)[:, off]
    true_positive = (predicted_same & target_same).sum().float()
    precision = true_positive / predicted_same.sum().clamp(min=1).float()
    recall = true_positive / target_same.sum().clamp(min=1).float()
    return (2 * precision * recall / (precision + recall).clamp(min=1e-6)).detach()


@torch.no_grad()
def _partition_scores(group_id: torch.Tensor, scene_id: torch.Tensor, prefix: str) -> dict:
    """Pair agreement, ARI, exact partition rate, and scene-count error of predicted groups."""
    num_frames = group_id.shape[-1]
    off = ~torch.eye(num_frames, device=group_id.device, dtype=torch.bool)
    predicted_same = group_id[:, :, None] == group_id[:, None, :]
    target_same = scene_id[:, :, None] == scene_id[:, None, :]
    rows = range(group_id.shape[0])
    ari = [adjusted_rand_index(group_id[row].cpu(), scene_id[row].cpu()) for row in rows]
    count_error = [
        abs(int(group_id[row].unique().numel()) - int(scene_id[row].unique().numel())) for row in rows
    ]
    exact = [float((predicted_same[row] == target_same[row]).all()) for row in rows]
    return {
        f"{prefix}group_agreement": (predicted_same == target_same)[:, off].float().mean(),
        f"{prefix}ari": torch.tensor(sum(ari) / len(ari)),
        f"{prefix}exact_partition": torch.tensor(sum(exact) / len(exact)),
        f"{prefix}scene_count_error": torch.tensor(sum(count_error) / len(count_error)),
    }
