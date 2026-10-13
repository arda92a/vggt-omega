"""Which metrics go to wandb and the console, and how they are named and printed.

Training logs the grouping scores and the loss. Validation logs the same grouping scores on held-out
scenes, and compares three systems on pose and depth:

- predicted: inference, with the groups the model found.
- oracle:    the same model given the true groups. The gap to `predicted` is what clustering errors cost.
- plain:     no isolation, all frames in one group. The gap to `oracle` is what mixing scenes costs.
"""

from typing import Mapping, Optional

# Grouping scores, all in [0, 1]. See vggt_omega.utils.scenes.partition_scores.
GROUPING = ("ari", "perfect_bags", "pair_acc", "count_acc", "separated", "together")
SYSTEMS = ("predicted", "oracle", "plain")
# (chart name, metric key). One chart overlays the three systems.
COMPARED = (("pose_auc30", "auc_30"), ("depth_abs_rel", "abs_rel"))

_TRAIN_RENAMES = {"loss_objective": "loss", "Auc_30": "auc_30"}

_LABELS = {
    "ari": ("ARI", "{:.3f}"),
    "perfect_bags": ("bags split exactly right", "{:.1%}"),
    "pair_acc": ("pairs right", "{:.1%}"),
    "count_acc": ("scene count right", "{:.1%}"),
    "separated": ("different scenes separated", "{:.1%}"),
    "together": ("same scene kept together", "{:.1%}"),
}


def train_scalars(averages: Mapping[str, float]) -> dict[str, float]:
    """Epoch averages of the training scalars, under `train/`."""
    return {f"train/{_TRAIN_RENAMES.get(name, name)}": value for name, value in averages.items()}


def val_scalars(metrics: Mapping[str, float]) -> dict[str, float]:
    """The validation numbers worth a chart of their own, all from the `predicted` system."""
    scalars = {}
    for key in GROUPING:
        if f"predicted/{key}" in metrics:
            scalars[f"val/{key}"] = metrics[f"predicted/{key}"]
    for key in ("ari", "pair_acc"):
        if f"predicted/route_{key}" in metrics:
            scalars[f"val/route_{key}"] = metrics[f"predicted/route_{key}"]
    for key in ("auc_30", "rra_15", "rta_15", "abs_rel"):
        if f"predicted/{key}" in metrics:
            scalars[f"val/{key}"] = metrics[f"predicted/{key}"]
    if "stats/num_bags" in metrics:
        scalars["val/num_bags"] = metrics["stats/num_bags"]
    return scalars


def val_comparisons(metrics: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """Per chart, the value of each system. Systems that were not scored are left out."""
    return {
        chart: {system: metrics[f"{system}/{key}"] for system in SYSTEMS if f"{system}/{key}" in metrics}
        for chart, key in COMPARED
    }


def format_epoch(
    epoch: int,
    max_epochs: int,
    train: Optional[Mapping[str, float]],
    val: Optional[Mapping[str, float]],
    train_seconds: float = 0.0,
    val_seconds: float = 0.0,
    eta_seconds: float = 0.0,
    title: Optional[str] = None,
) -> str:
    """One multi-line block for the whole epoch, so lines of different phases never interleave."""
    title = title or f"epoch {epoch + 1}/{max_epochs}"
    header = f"==== {title} "
    if train_seconds:
        header += f"| train {_clock(train_seconds)} | val {_clock(val_seconds)} | time left {_clock(eta_seconds)} "
    lines = [header.ljust(86, "=")]
    if train:
        lines.append("train  " + _join(_train_losses(train)))
        lines.append("       " + _join(_grouping_parts(train, "")))
    if val:
        lines.append("val    " + _join(_grouping_parts(val, "predicted/")))
        route = _grouping_parts(val, "predicted/route_", keys=("ari", "pair_acc"))
        if route:
            lines.append("       route head: " + _join(route))
        comparisons = val_comparisons(val)
        for chart, label in (("pose_auc30", "pose AUC@30 (up)"), ("depth_abs_rel", "depth AbsRel (down)")):
            if comparisons[chart]:
                lines.append(f"       {label:<20}" + _join([f"{name} {value:.3f}" for name, value in comparisons[chart].items()]))
        if "predicted/rra_15" in val:
            lines.append(
                "       "
                + _join([f"rotation acc@15 {val['predicted/rra_15']:.1%}", f"translation acc@15 {val['predicted/rta_15']:.1%}"])
            )
        if "stats/num_bags" in val:
            lines.append(f"       ({int(val['stats/num_bags'])} held-out bags, {val.get('stats/num_scenes', 0):.1f} scenes each)")
    return "\n" + "\n".join(lines)


def format_data_summary(
    roots: list[str],
    splits: Mapping[str, Mapping[str, tuple[int, int]]],
    bags: Mapping[str, Mapping],
) -> str:
    """What the run trains and validates on. `splits` maps split -> source -> (scenes, frames)."""
    lines = ["==== data ".ljust(86, "=")]
    lines += [f"root   {root}" for root in roots]
    for split, sources in splits.items():
        scenes = sum(count for count, _ in sources.values())
        frames = sum(total for _, total in sources.values())
        detail = " | ".join(f"{name}: {count} scenes, {total:,} frames" for name, (count, total) in sorted(sources.items()))
        lines.append(f"{split:<6} {scenes} scenes, {frames:,} frames  ({detail})")
    for split, conf in bags.items():
        if not conf:
            continue
        low, high = conf.get("num_scenes_range", (conf.get("num_scenes", 2),) * 2)
        text = f"{low}-{high} scenes per bag"
        if conf.get("frames_range"):
            text += f", {conf['frames_range'][0]}-{conf['frames_range'][1]} frames per scene"
        if conf.get("max_total_frames"):
            text += f", at most {conf['max_total_frames']} frames"
        text += f", {float(conf.get('same_dataset_prob', 0.0)):.0%} of bags from one dataset"
        lines.append(f"bags   {split}: {text}")
    return "\n" + "\n".join(lines)


def _train_losses(train: Mapping[str, float]) -> list[str]:
    parts = [f"loss {train['loss_objective']:.3f}"] if "loss_objective" in train else []
    for key, label in (("loss_camera", "camera loss"), ("loss_depth", "depth loss"), ("loss_point", "point loss")):
        if key in train:
            parts.append(f"{label} {train[key]:.3f}")
    if "gt_group_prob" in train:
        parts.append(f"true groups used in {train['gt_group_prob']:.0%} of steps")
    return parts


def _grouping_parts(values: Mapping[str, float], prefix: str, keys=GROUPING) -> list[str]:
    return [
        f"{_LABELS[key][0]} {_LABELS[key][1].format(values[prefix + key])}"
        for key in keys
        if prefix + key in values
    ]


def _join(parts: list[str]) -> str:
    return " | ".join(parts)


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"
