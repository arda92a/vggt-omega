"""Which metrics go to wandb and the console, and how they are named and printed.

The task is grouping, so two numbers carry it: ARI and the share of bags split exactly right. Validation
adds the same two per data source (how well scenes of one source are told apart). Stage 2 also reports
pose and depth for three systems:

- predicted: inference, with the groups the model found.
- oracle:    the same model given the true groups. The gap to `predicted` is what clustering errors cost.
- plain:     no isolation, all frames in one group. The gap to `oracle` is what mixing scenes costs.
"""

from typing import Mapping, Optional

SYSTEMS = ("predicted", "oracle", "plain")
# (chart name, metric key). One chart overlays the three systems.
COMPARED = (("pose_auc30", "auc_30"), ("depth_abs_rel", "abs_rel"))

# Training scalars that reach wandb, and the name they get there.
_WANDB_TRAIN = {
    "loss_objective": "loss",
    "ari": "ari",
    "perfect_bags": "perfect_bags",
    "loss_camera": "loss_camera",
    "loss_depth": "loss_depth",
    "loss_point": "loss_point",
}


def train_scalars(averages: Mapping[str, float]) -> dict[str, float]:
    """Epoch averages of the few training scalars worth a chart, under `train/`."""
    return {f"train/{new}": averages[old] for old, new in _WANDB_TRAIN.items() if old in averages}


def source_names(metrics: Mapping[str, float]) -> list[str]:
    """Data sources that have a within-source score, for example eth3d."""
    names = []
    for key in metrics:
        name = key[len("predicted/") : -len("_ari")]
        if key.startswith("predicted/") and key.endswith("_ari") and name and name != "route":
            names.append(name)
    return sorted(names)


def val_scalars(metrics: Mapping[str, float]) -> dict[str, float]:
    """Validation charts: grouping overall and per source, plus pose and depth when they were scored."""
    scalars = {}
    for key in ("ari", "perfect_bags"):
        if f"predicted/{key}" in metrics:
            scalars[f"val/{key}"] = metrics[f"predicted/{key}"]
    for source in source_names(metrics):
        scalars[f"val/{source}_ari"] = metrics[f"predicted/{source}_ari"]
    for key in ("auc_30", "abs_rel"):
        if f"predicted/{key}" in metrics:
            scalars[f"val/{key}"] = metrics[f"predicted/{key}"]
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
        lines.append("train  " + _join(_train_parts(train)))
    if val:
        lines.append("val    " + _join(_grouping_parts(val, "predicted/")))
        sources = [f"{name} ARI {val[f'predicted/{name}_ari']:.3f}" for name in source_names(val)]
        if sources:
            lines.append("       scenes of one source told apart: " + _join(sources))
        comparisons = val_comparisons(val)
        for chart, label in (("pose_auc30", "pose AUC@30 (up)"), ("depth_abs_rel", "depth AbsRel (down)")):
            if comparisons[chart]:
                lines.append(f"       {label:<20}" + _join([f"{name} {value:.3f}" for name, value in comparisons[chart].items()]))
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


_LABELS = {"ari": ("ARI", "{:.3f}"), "perfect_bags": ("bags split exactly right", "{:.1%}")}


def _train_parts(train: Mapping[str, float]) -> list[str]:
    parts = [f"loss {train['loss_objective']:.3f}"] if "loss_objective" in train else []
    parts += _grouping_parts(train, "")
    for key, label in (("loss_camera", "camera loss"), ("loss_depth", "depth loss"), ("loss_point", "point loss")):
        if key in train:
            parts.append(f"{label} {train[key]:.3f}")
    if "gt_group_prob" in train:
        parts.append(f"true groups used in {train['gt_group_prob']:.0%} of steps")
    return parts


def _grouping_parts(values: Mapping[str, float], prefix: str) -> list[str]:
    return [
        f"{name} {fmt.format(values[prefix + key])}"
        for key, (name, fmt) in _LABELS.items()
        if prefix + key in values
    ]


def _join(parts: list[str]) -> str:
    return " | ".join(parts)


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"
