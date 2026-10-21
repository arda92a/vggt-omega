"""Loss and the three metrics reported every epoch.

  AUROC        how well same-sequence pairs score above different-sequence pairs (no threshold, 0.5 is chance)
  ARI          how well the predicted groups of a bag match its sequences (0 is chance, 1 is perfect)
  Perfect bag  share of bags grouped with no mistake at all
"""

import torch
import torch.nn.functional as F

from vggt_omega.models.multiscene import scene_groups


def pair_targets(labels: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """same (B, N, N): 1 where two frames share a sequence. valid (B, N, N): real pairs i < j only."""
    same = (labels[:, :, None] == labels[:, None, :]).float()
    valid = mask[:, :, None] & mask[:, None, :]
    valid &= torch.ones_like(valid[0]).triu(1).bool()
    return same, valid


def affinity_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on every pair, positives and negatives weighted equally."""
    same, valid = pair_targets(labels, mask)
    losses = F.binary_cross_entropy_with_logits(logits, same, reduction="none")
    positive, negative = valid & (same > 0), valid & (same == 0)
    return 0.5 * (losses[positive].mean() + losses[negative].mean())


def auroc(scores: torch.Tensor, same: torch.Tensor) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    same = same.bool()
    _, inverse, counts = scores.unique(return_inverse=True, return_counts=True)
    ranks = (counts.cumsum(0) - (counts - 1) / 2)[inverse]
    positive, negative = int(same.sum()), int((~same).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    return float((ranks[same].sum() - positive * (positive + 1) / 2) / (positive * negative))


def adjusted_rand_index(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    table = torch.zeros(int(predicted.max()) + 1, int(truth.max()) + 1)
    table.index_put_((predicted, truth), torch.ones(len(truth)), accumulate=True)

    def pairs(x: torch.Tensor) -> torch.Tensor:
        return (x * (x - 1) / 2).sum()

    index, rows, cols, total = pairs(table), pairs(table.sum(1)), pairs(table.sum(0)), pairs(torch.tensor([len(truth)]))
    expected = rows * cols / total
    best = (rows + cols) / 2
    return 1.0 if best == expected else float((index - expected) / (best - expected))


class EpochMetrics:
    """Collects one epoch of predictions, then returns loss, AUROC, ARI and perfect-bag ratio."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.losses, self.scores, self.same, self.aris, self.perfect = [], [], [], [], []

    @torch.no_grad()
    def add(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
        logits, labels, mask = logits.detach().float().cpu(), labels.cpu(), mask.cpu()
        same, valid = pair_targets(labels, mask)
        self.losses.append(loss.item())
        self.scores.append(logits[valid])
        self.same.append(same[valid])
        for bag_logits, bag_labels, bag_mask in zip(logits, labels, mask):
            n = int(bag_mask.sum())
            groups = scene_groups(bag_logits[:n, :n].sigmoid()[None], self.threshold)[0]
            truth = bag_labels[:n]
            self.aris.append(adjusted_rand_index(groups, truth))
            self.perfect.append(float(torch.equal(groups[:, None] == groups[None], truth[:, None] == truth[None])))

    def compute(self) -> dict[str, float]:
        mean = lambda values: sum(values) / max(len(values), 1)  # noqa: E731
        return {
            "loss": mean(self.losses),
            "auroc": auroc(torch.cat(self.scores), torch.cat(self.same)),
            "ari": mean(self.aris),
            "perfect_bag": mean(self.perfect),
        }
