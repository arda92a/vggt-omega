"""Check a unified dataset and split its scenes into train.txt and val.txt.

    uv run python training/dataset_preparation/split_sequences.py /path/to/merged --val-per-source 4

A scene is one folder under the root. The source is the part of its name before the first
underscore (eth3d_courtyard -> eth3d), and every source gives at least --val-per-source scenes
to validation, so both sources are checked on held-out scenes. Scenes of one family
(replica_apartment_0 .. _2, eth3d_relief and eth3d_relief_2) stay on the same side. The split is by
scene, never by frame, and is fixed by --seed. The lists are written next to the scenes.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

MIN_FRAMES = 3


def check_scene(scene: Path) -> list[str]:
    """Return what is wrong with one scene folder. An empty list means the loader can read it."""
    problems = []
    names_path = scene / "image_names.json"
    if not names_path.is_file():
        return ["image_names.json is missing"]
    names = json.loads(names_path.read_text())
    if len(names) < MIN_FRAMES:
        problems.append(f"only {len(names)} frames, need {MIN_FRAMES}")
    for filename, shape in (("cam_from_worlds.npy", (3, 4)), ("intrinsics.npy", (3, 3))):
        path = scene / filename
        if not path.is_file():
            problems.append(f"{filename} is missing")
            continue
        array = np.load(path, mmap_mode="r")
        if array.shape != (len(names), *shape):
            problems.append(f"{filename} has shape {array.shape}, expected {(len(names), *shape)}")
    missing_images = [name for name in names if not (scene / "images" / name).is_file()]
    missing_depths = [name for name in names if not (scene / "depths" / f"{Path(name).stem}.exr").is_file()]
    if missing_images:
        problems.append(f"{len(missing_images)} images missing, first {missing_images[0]}")
    if missing_depths:
        problems.append(f"{len(missing_depths)} depths missing, first {missing_depths[0]}")
    return problems


def family(name: str) -> str:
    """Scene name without its source and trailing index: eth3d_relief_2 -> relief."""
    rest = name.partition("_")[2]
    return re.sub(r"_?\d+$", "", rest) or rest


def split(scenes: list[str], val_per_source: int, seed: int) -> tuple[list[str], list[str]]:
    """Hold out whole families, so frl_apartment_0 and frl_apartment_3 never sit on both sides.

    Families are added to val until the source has at least `val_per_source` val scenes.
    """
    by_source: dict[str, list[str]] = {}
    for name in scenes:
        by_source.setdefault(name.split("_")[0], []).append(name)
    train, val = [], []
    for source, names in sorted(by_source.items()):
        by_family: dict[str, list[str]] = {}
        for name in names:
            by_family.setdefault(family(name), []).append(name)
        if len(names) <= val_per_source or len(by_family) < 2:
            raise SystemExit(
                f"Source '{source}' has {len(names)} scenes in {len(by_family)} families, "
                f"which cannot give {val_per_source} validation scenes and keep a training family."
            )
        families = sorted(by_family)
        random.Random(seed).shuffle(families)
        # Small families first, so one six-scene family does not swallow the whole val budget.
        families.sort(key=lambda name: len(by_family[name]))
        source_val = []
        for name in families[:-1]:
            if len(source_val) >= val_per_source:
                break
            source_val += by_family[name]
        val += source_val
        train += [name for name in names if name not in source_val]
    return sorted(train), sorted(val)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="unified dataset root, one folder per scene")
    parser.add_argument("--val-per-source", type=int, default=4, help="validation scenes taken from each source")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scenes = sorted(path.name for path in args.root.iterdir() if path.is_dir())
    if not scenes:
        raise SystemExit(f"No scene folders under {args.root}")
    broken = {name: check_scene(args.root / name) for name in scenes}
    broken = {name: problems for name, problems in broken.items() if problems}
    if broken:
        for name, problems in broken.items():
            print(f"{name}: " + "; ".join(problems))
        raise SystemExit(f"{len(broken)} of {len(scenes)} scenes cannot be loaded. Fix them, then run again.")

    train, val = split(scenes, args.val_per_source, args.seed)
    (args.root / "train.txt").write_text("\n".join(train) + "\n")
    (args.root / "val.txt").write_text("\n".join(val) + "\n")
    print(f"{len(train)} train scenes, {len(val)} val scenes")
    print("val: " + ", ".join(val))


if __name__ == "__main__":
    sys.exit(main())
