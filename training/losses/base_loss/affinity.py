"""Affinity loss and the grouping metrics logged with it.

The loss is a class-balanced BCE over off-diagonal pairs. With S scenes only 1/S of the
pairs are positive, and the diagonal is always 1, so a plain mean rewards "everything is
the same scene". The route head (middle layer) and the final head are supervised
separately, and optionally the final head again after scene attention.

The metrics come from the hard groups the model would use, not from the soft affinity.
"""

import torch
import torch.nn.functional as F

from vggt_omega.utils.scenes import partition_scores


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
    }
    if include_output:
        if "affinity_output_logits" not in predictions:
            raise KeyError("include_output requires predictions['affinity_output_logits']")
        output_loss = balanced_bce(predictions["affinity_output_logits"], target)
        total = total + output_loss
        logs["loss_affinity_output"] = output_loss.detach()
    logs["loss_affinity"] = total

    if "scene_id" in batch:
        logs.update(_group_metrics(predictions["group_id"], batch["scene_id"], prefix=""))
        logs.update(_group_metrics(predictions["route_group_id"], batch["scene_id"], prefix="route_"))
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


@torch.no_grad()
def _group_metrics(group_id: torch.Tensor, scene_id: torch.Tensor, prefix: str) -> dict:
    """Bag-averaged partition scores. A score a bag cannot define is left out of its average."""
    rows = [partition_scores(group_id[row], scene_id[row]) for row in range(group_id.shape[0])]
    metrics = {}
    for name in {key for row in rows for key in row}:
        values = [row[name] for row in rows if name in row]
        metrics[f"{prefix}{name}"] = torch.tensor(sum(values) / len(values))
    return metrics
