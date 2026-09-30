"""Binary affinity loss and the two numbers worth watching while it trains."""

import torch
import torch.nn.functional as F


def compute_affinity_loss(predictions, batch, include_output: bool = False) -> dict:
    """BCE between predicted affinity and the ground-truth scene adjacency.

    The diagonal is ignored in the reported F1. It is always 1, so it would
    make a random head look better than it is. `include_output` adds the same
    loss on affinity read after scene attention, which is what trains that block.
    """
    if "affinity" not in predictions:
        raise KeyError("Affinity loss requires predictions['affinity']. Enable model.multiscene.")
    if "affinity" not in batch:
        raise KeyError(
            "Affinity loss requires batch['affinity']. Enable data.train.mixed_scene with num_scenes >= 2."
        )

    target = batch["affinity"].to(dtype=predictions["affinity"].dtype)
    loss = _bce(predictions["affinity"], target)
    logs = {
        "loss_affinity": loss,
        "affinity_f1": _off_diagonal_f1(predictions["affinity"], target),
    }
    if include_output:
        if "affinity_output" not in predictions:
            raise KeyError("include_output requires predictions['affinity_output']")
        output_loss = _bce(predictions["affinity_output"], target)
        logs["loss_affinity"] = logs["loss_affinity"] + output_loss
        logs["affinity_output_f1"] = _off_diagonal_f1(predictions["affinity_output"], target)

    if "group_id" in predictions and "scene_id" in batch:
        logs["group_agreement"] = _group_agreement(predictions["group_id"], batch["scene_id"])
    return logs


def _bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(prediction, target)


def _off_diagonal_f1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_frames = prediction.shape[-1]
    eye = torch.eye(num_frames, device=prediction.device, dtype=torch.bool)
    off = ~eye
    predicted_same = (prediction >= 0.5)[:, off]
    target_same = (target >= 0.5)[:, off]
    true_positive = (predicted_same & target_same).sum().float()
    precision = true_positive / predicted_same.sum().clamp(min=1).float()
    recall = true_positive / target_same.sum().clamp(min=1).float()
    return 2 * precision * recall / (precision + recall).clamp(min=1e-6)


def _group_agreement(group_id: torch.Tensor, scene_id: torch.Tensor) -> torch.Tensor:
    """Fraction of off-diagonal pairs whose same-group decision matches the label."""
    num_frames = group_id.shape[-1]
    eye = torch.eye(num_frames, device=group_id.device, dtype=torch.bool)
    predicted_same = group_id[:, :, None] == group_id[:, None, :]
    target_same = scene_id[:, :, None] == scene_id[:, None, :]
    return (predicted_same == target_same)[:, ~eye].float().mean()
