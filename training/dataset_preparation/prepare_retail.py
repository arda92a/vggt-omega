"""Group a flat folder of retail frames into one scene folder per sequence.

Input:   retail_data/images/seq001_01.jpg, seq001_02.jpg, ... (other files such as *.json are ignored)
Output:  RETAIL_DIR/retail_seq001/images/seq001_01.jpg ... plus image_names.json per scene

The scenes have no depth or pose, so they only train grouping (stage 1, config multiscene_stage1_retail).

    uv run python training/dataset_preparation/prepare_retail.py retail_data RETAIL_DIR
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

FRAME = re.compile(r"^(?P<seq>.+)_(?P<index>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def group_frames(image_dir: Path) -> dict[str, list[str]]:
    sequences = defaultdict(list)
    for path in sorted(image_dir.iterdir()):
        match = FRAME.match(path.name)
        if match:
            sequences[match["seq"]].append((int(match["index"]), path.name))
    return {seq: [name for _, name in sorted(frames)] for seq, frames in sorted(sequences.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="folder that contains images/ (or the images folder itself)")
    parser.add_argument("dst", type=Path, nargs="?", default=Path("RETAIL_DIR"), help="output root (default: RETAIL_DIR)")
    parser.add_argument("--prefix", default="retail", help="scene folder prefix, which names the data source")
    parser.add_argument("--min-frames", type=int, default=3, help="skip sequences with fewer frames")
    parser.add_argument("--copy", action="store_true", help="copy the images instead of symlinking them")
    args = parser.parse_args()

    image_dir = args.src / "images" if (args.src / "images").is_dir() else args.src
    sequences = group_frames(image_dir)
    if not sequences:
        raise SystemExit(f"No frames named <sequence>_<number>.jpg found in {image_dir}")

    kept, skipped = 0, []
    for seq, names in sequences.items():
        if len(names) < args.min_frames:
            skipped.append(f"{seq} ({len(names)})")
            continue
        scene = args.dst / f"{args.prefix}_{seq}"
        (scene / "images").mkdir(parents=True, exist_ok=True)
        for name in names:
            target = scene / "images" / name
            if target.exists() or target.is_symlink():
                target.unlink()
            if args.copy:
                shutil.copy2(image_dir / name, target)
            else:
                os.symlink((image_dir / name).resolve(), target)
        (scene / "image_names.json").write_text(json.dumps(names))
        kept += 1

    counts = [len(names) for names in sequences.values() if len(names) >= args.min_frames]
    print(f"{kept} scenes written to {args.dst} ({min(counts)}-{max(counts)} frames, {sum(counts)} in total)")
    if skipped:
        print(f"skipped {len(skipped)} sequences with fewer than {args.min_frames} frames: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
