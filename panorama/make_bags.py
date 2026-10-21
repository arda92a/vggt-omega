"""Build an evaluation set of bags: frames from several scenes, mixed, with the true scene of every frame kept.

Every bag is a folder of neutrally named frames (f00.jpg, f01.jpg, ...) in shuffled order, so neither the order nor
the file name gives the scene away. bags.json holds the truth, index.html shows every bag.

    uv run python panorama/make_bags.py --config panorama/bags.yaml
    uv run python panorama/make_bags.py --num-bags 30 --scenes 2 4 --frames 3 6 --out bags_eval
    uv run python panorama/make_bags.py --scene-sampling partition --frames all --max-frames 24

scene_sampling partition packs every scene into exactly one bag with all of its frames: no repeats, nothing dropped
except scenes that fit no bag. num_bags and the frames range are ignored, the bag count follows from the packing.

Scenes are folders with an images/ subfolder, for example RETAIL_DIR/retail_12/images/*.jpg. The start of the folder
name up to the first underscore is the scene's source (retail, eth3d, replica). Settings come from the command line,
else from the config file, else from the defaults below. The resolved settings are written to the output folder.
"""

import argparse
import html
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
OWN_FILES = {"bags", "thumbs", "bags.json", "config.json", "index.html", ".DS_Store"}
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#16a5a5", "#d4a017", "#f032e6", "#7f7f00", "#808080"]

DEFAULTS = {
    "roots": ["RETAIL_DIR", "MERGED_DIR"],  # folders of scene folders, missing ones are skipped
    "sources": None,                         # only these sources, for example [retail]. None: all
    "out": "bags_eval",
    "num_bags": 50,                         # ignored by partition
    "scenes": [2, 4],                        # scenes per bag, min and max (partition: the limits of the packing)
    "frames": [3, 6],                        # frames per scene, min and max, or "all" (partition always uses all)
    "max_frames": 16,                        # cap on the whole bag
    "same_source_prob": 0.5,                 # chance that all scenes of a bag come from one source
    "scene_sampling": "balanced",            # balanced: least used first, random: uniform, partition: each scene once
    "frame_pick": "random",                  # random frames of a scene, or consecutive: a run of neighbouring frames
    "seed": 0,
    "link": "copy",                         # how bag images are stored: copy, symlink or none
    "thumb_size": 256,
}


def natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return ROOT / candidate


def read_scenes(roots: list[str], sources: list[str] | None, min_frames: int) -> dict[str, list[str]]:
    scenes = {}
    for root in roots:
        folder = resolve(root)
        if not folder.is_dir():
            continue
        for scene in sorted(p for p in folder.iterdir() if (p / "images").is_dir()):
            if sources and scene.name.split("_")[0] not in sources:
                continue
            names = sorted((n for n in os.listdir(scene / "images") if n.lower().endswith(IMAGE_TYPES)), key=natural_key)
            if len(names) >= min_frames:
                scenes[scene.name] = [os.path.realpath(scene / "images" / n) for n in names]
    return scenes


class BagSampler:
    def __init__(self, scenes: dict[str, list[str]], settings: dict) -> None:
        self.scenes = scenes
        self.s = settings
        self.rng = random.Random(settings["seed"])
        self.usage = Counter()
        self.by_source: dict[str, list[str]] = {}
        for name in scenes:
            self.by_source.setdefault(name.split("_")[0], []).append(name)

    def _pick_scenes(self, pool: list[str], count: int) -> list[str]:
        if self.s["scene_sampling"] == "balanced":
            order = sorted(pool, key=lambda name: (self.usage[name], self.rng.random()))
            return order[:count]
        return self.rng.sample(pool, count)

    def _pick_frames(self, frames: list[str], count: int) -> list[str]:
        if self.s["frame_pick"] == "consecutive":
            start = self.rng.randint(0, len(frames) - count)
            return frames[start : start + count]
        return sorted(self.rng.sample(frames, count), key=lambda path: natural_key(os.path.basename(path)))

    def bag(self) -> list[tuple[str, str]]:
        """(scene, image path) pairs in shuffled order."""
        scene_min, scene_max = self.s["scenes"]
        frame_min, frame_max = self.s["frames"]
        count = self.rng.randint(scene_min, scene_max)
        count = min(count, len(self.scenes), self.s["max_frames"] // frame_min)

        pool = list(self.scenes)
        if self.rng.random() < self.s["same_source_prob"]:
            wide = [names for names in self.by_source.values() if len(names) >= count]
            if wide:
                pool = self.rng.choice(wide)
        chosen = self._pick_scenes(pool, count)

        sizes = [self.rng.randint(frame_min, min(frame_max, len(self.scenes[name]))) for name in chosen]
        while sum(sizes) > self.s["max_frames"]:
            sizes[sizes.index(max(sizes))] -= 1
        frames = []
        for name, size in zip(chosen, sizes):
            self.usage[name] += 1
            frames += [(name, path) for path in self._pick_frames(self.scenes[name], size)]
        self.rng.shuffle(frames)
        return frames


def pack_scenes(sizes: dict[str, int], cap: int, scene_min: int, scene_max: int, rng: random.Random):
    """First fit over shuffled scenes. Returns (bags as lists of scene names, scenes that fit nowhere)."""
    names = list(sizes)
    rng.shuffle(names)
    bags: list[list[str]] = []
    load: list[int] = []
    left = [n for n in names if sizes[n] > cap]
    for name in (n for n in names if sizes[n] <= cap):
        for i, bag in enumerate(bags):
            if len(bag) < scene_max and load[i] + sizes[name] <= cap:
                bag.append(name)
                load[i] += sizes[name]
                break
        else:
            bags.append([name])
            load.append(sizes[name])

    short = [b for b in bags if len(b) < scene_min]
    bags = [b for b in bags if len(b) >= scene_min]
    pending = [n for b in short for n in b]
    for name in pending:
        room = [b for b in bags if len(b) < scene_max and sum(sizes[s] for s in b) + sizes[name] <= cap]
        if room:
            min(room, key=lambda b: sum(sizes[s] for s in b)).append(name)
            continue
        # Take one scene from a bag that can spare it and start a two-scene bag with it.
        donors = [(b, s) for b in bags if len(b) > scene_min for s in b if sizes[s] + sizes[name] <= cap]
        if scene_min == 2 and donors:
            bag, donated = rng.choice(donors)
            bag.remove(donated)
            bags.append([name, donated])
        else:
            left.append(name)
    rng.shuffle(bags)
    return bags, left


def partition_bags(scenes: dict[str, list[str]], settings: dict):
    """Every scene in exactly one bag with all its frames. Returns (bags of (scene, path) pairs, left out scenes)."""
    rng = random.Random(settings["seed"])
    sizes = {name: len(paths) for name, paths in scenes.items()}
    scene_min, scene_max = settings["scenes"]
    cap = settings["max_frames"]

    sources = sorted({name.split("_")[0] for name in scenes})
    groups = [list(scenes)]
    if len(sources) > 1 and settings["same_source_prob"] > 0:
        order = list(scenes)
        rng.shuffle(order)
        same = [n for n in order if rng.random() < settings["same_source_prob"]]
        mixed = [n for n in order if n not in set(same)]
        groups = [[n for n in same if n.split("_")[0] == s] for s in sources] + [mixed]

    bag_names, left = [], []
    for group in groups:
        if not group:
            continue
        packed, rest = pack_scenes({n: sizes[n] for n in group}, cap, scene_min, scene_max, rng)
        bag_names += packed
        left += rest
    rng.shuffle(bag_names)

    bags = []
    for names in bag_names:
        frames = [(name, path) for name in names for path in scenes[name]]
        rng.shuffle(frames)
        bags.append(frames)
    return bags, left


def thumbnail(source: str, target: Path, size: int) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.draft("RGB", (size * 2, size * 2))
        image = image.convert("RGB")
        image.thumbnail((size, size))
        image.save(target, quality=82)


def store(source: str, target: Path, mode: str) -> None:
    if mode == "none":
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        os.symlink(source, target)


def build(settings: dict, scenes: dict[str, list[str]], out: Path) -> dict:
    left_out = []
    if settings["scene_sampling"] == "partition":
        frame_lists, left = partition_bags(scenes, settings)
        left_out = [{"scene": name, "frames": len(scenes[name])} for name in sorted(left)]
    else:
        sampler = BagSampler(scenes, settings)
        frame_lists = [sampler.bag() for _ in range(settings["num_bags"])]
    usage = Counter(name for frames in frame_lists for name in {n for n, _ in frames})
    manifest_bags = []
    for number, frames in enumerate(frame_lists, 1):
        bag_id = f"bag_{number:04d}"
        order = list(dict.fromkeys(name for name, _ in frames))
        entries = []
        for index, (scene, source) in enumerate(frames):
            suffix = Path(source).suffix.lower()
            file = f"bags/{bag_id}/f{index:02d}{suffix}"
            store(source, out / file, settings["link"])
            thumb = f"thumbs/{scene}/{Path(source).stem}.jpg"
            thumbnail(source, out / thumb, settings["thumb_size"])
            entries.append({"file": file, "thumb": thumb, "source": source, "scene": scene, "label": order.index(scene)})
        sizes = Counter(e["label"] for e in entries)
        same = sum(n * (n - 1) // 2 for n in sizes.values())
        total = len(entries) * (len(entries) - 1) // 2
        manifest_bags.append(
            {
                "id": bag_id,
                "num_scenes": len(order),
                "num_frames": len(entries),
                "same_scene_pairs": same,
                "different_scene_pairs": total - same,
                "scenes": order,
                "frames": entries,
            }
        )
    return {
        "settings": settings,
        "usage": dict(usage),
        "num_scenes_available": len(scenes),
        "left_out": left_out,
        "bags": manifest_bags,
    }


def summary(manifest: dict) -> str:
    bags = manifest["bags"]
    count = Counter(b["num_scenes"] for b in bags)
    used = manifest["usage"]
    lines = [
        f"{len(bags)} bags from {manifest['num_scenes_available']} scenes ({len(used)} used, each 1-{max(used.values())} times)",
        "scenes per bag: " + ", ".join(f"{k}: {v}" for k, v in sorted(count.items())),
        f"frames per bag: {sum(b['num_frames'] for b in bags) / len(bags):.1f} on average, "
        f"same-scene pairs {sum(b['same_scene_pairs'] for b in bags)}, different-scene pairs {sum(b['different_scene_pairs'] for b in bags)}",
    ]
    if manifest["settings"]["scene_sampling"] == "partition":
        frames = sum(b["num_frames"] for b in bags)
        lines.append(f"frames included: {frames}")
    left = manifest.get("left_out") or []
    if left:
        lines.append(
            f"left out, they fit no bag: {len(left)} scenes, {sum(item['frames'] for item in left)} frames "
            f"({', '.join(item['scene'] for item in left)})"
        )
    return "\n".join(lines)


def write_html(manifest: dict, out: Path) -> None:
    cards = []
    for bag in manifest["bags"]:
        thumbs = "".join(
            f'<figure class="thumb" style="--c:{PALETTE[f["label"] % len(PALETTE)]};--o:{f["label"]}" '
            f'title="{html.escape(f["scene"])} | {html.escape(os.path.basename(f["source"]))}">'
            f'<img src="{html.escape(f["thumb"])}" loading="lazy"><figcaption>{Path(f["file"]).stem}</figcaption></figure>'
            for f in bag["frames"]
        )
        counts = Counter(f["label"] for f in bag["frames"])
        legend = "".join(
            f'<span class="tag" style="--c:{PALETTE[i % len(PALETTE)]}">{html.escape(name)} &middot; {counts[i]}</span>'
            for i, name in enumerate(bag["scenes"])
        )
        cards.append(
            f'<section class="bag"><h2>{bag["id"]} <small>{bag["num_scenes"]} scenes, {bag["num_frames"]} frames, '
            f'{bag["same_scene_pairs"]} same / {bag["different_scene_pairs"]} different pairs</small></h2>'
            f'<div class="frames">{thumbs}</div><div class="legend">{legend}</div></section>'
        )
    settings = html.escape(json.dumps(manifest["settings"]))
    page = f"""<!doctype html><meta charset="utf-8"><title>Evaluation bags</title>
<style>
body{{font:14px system-ui,sans-serif;margin:20px;background:#fafafa;color:#222}}
.bag{{background:#fff;border:1px solid #d8dde3;border-radius:8px;padding:10px 14px;margin:14px 0}}
h2{{font-size:15px;margin:0 0 8px}} h2 small{{font-weight:400;color:#666;margin-left:8px}}
.frames{{display:flex;flex-wrap:wrap;gap:6px}}
.thumb{{margin:0;border:3px solid transparent;border-radius:4px;text-align:center;font-size:11px;color:#666}}
.thumb img{{display:block;height:170px}}
.truth .thumb{{border-color:var(--c);order:var(--o)}}
.legend{{display:none;margin-top:8px}} .truth .legend{{display:block}}
.tag{{display:inline-block;border:2px solid var(--c);border-radius:12px;padding:1px 9px;margin-right:6px;font-size:12px}}
.top{{position:sticky;top:0;background:#fafafa;padding:6px 0;z-index:1}}
</style>
<div class="top"><label><input type="checkbox" id="truth"> show true scenes (colours, frames grouped by scene)</label></div>
<pre>{html.escape(summary(manifest))}</pre><details><summary>settings</summary><code>{settings}</code></details>
{''.join(cards)}
<script>document.getElementById('truth').addEventListener('change',e=>document.body.classList.toggle('truth',e.target.checked));</script>
"""
    (out / "index.html").write_text(page)


def parse() -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with the settings below (same names, underscores)")
    parser.add_argument("--roots", nargs="+", help="folders of scene folders")
    parser.add_argument("--sources", nargs="+", help="keep only scenes of these sources")
    parser.add_argument("--out", help="output folder")
    parser.add_argument("--num-bags", type=int)
    parser.add_argument("--scenes", type=int, nargs=2, metavar=("MIN", "MAX"), help="scenes per bag")
    parser.add_argument("--frames", nargs="+", metavar="MIN MAX | all", help="frames per scene: two numbers, or all")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--same-source-prob", type=float)
    parser.add_argument("--scene-sampling", choices=["balanced", "random", "partition"])
    parser.add_argument("--frame-pick", choices=["random", "consecutive"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--link", choices=["symlink", "copy", "none"])
    parser.add_argument("--thumb-size", type=int)
    parser.add_argument("--overwrite", action="store_true", help="replace a non-empty output folder")
    args = parser.parse_args()

    settings = dict(DEFAULTS)
    if args.config:
        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise SystemExit(f"unknown settings in {args.config}: {sorted(unknown)}")
        settings.update(loaded)
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    settings["overwrite"] = args.overwrite
    frames = settings["frames"]
    if isinstance(frames, str) or (isinstance(frames, list) and len(frames) == 1 and str(frames[0]).lower() == "all"):
        frames = "all"
    elif len(frames) == 2:
        frames = [int(v) for v in frames]
    else:
        raise SystemExit("frames needs two numbers (min max) or all")
    settings["frames"] = frames
    return settings


def main() -> None:
    settings = parse()
    overwrite = settings.pop("overwrite")
    low, high = settings["scenes"]
    partition = settings["scene_sampling"] == "partition"
    if not (1 <= low <= high):
        raise SystemExit("scenes needs 1 <= min <= max")
    if partition:
        if settings["frames"] != "all":
            print("partition uses every frame of every scene, the frames range is ignored")
            settings["frames"] = "all"
        if high * 2 > settings["max_frames"] and low > 1:
            print("note: max_frames is small for the largest bags")
        frame_low = 1
    else:
        if settings["frames"] == "all":
            raise SystemExit("frames: all needs scene_sampling: partition")
        frame_low, frame_high = settings["frames"]
        if not (1 <= frame_low <= frame_high):
            raise SystemExit("frames needs 1 <= min <= max")
        if settings["max_frames"] < frame_low * 2:
            raise SystemExit("max_frames is too small for two scenes")

    scenes = read_scenes(settings["roots"], settings["sources"], frame_low)
    if len(scenes) < max(low, 2):
        raise SystemExit(f"Found {len(scenes)} usable scenes in {settings['roots']}. Need at least {max(low, 2)}.")
    print(f"{len(scenes)} scenes: " + ", ".join(f"{s} {n}" for s, n in sorted(Counter(n.split('_')[0] for n in scenes).items())))

    out = Path(settings["out"]).expanduser()
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out} is not empty. Pass --overwrite to replace it.")
        foreign = {p.name for p in out.iterdir()} - OWN_FILES
        if foreign:
            raise SystemExit(f"{out} holds files this script did not write ({sorted(foreign)[:3]}), refusing to delete it.")
        for _ in range(5):
            shutil.rmtree(out, ignore_errors=True)
            if not out.exists():
                break
        else:
            raise SystemExit(f"Could not delete {out}. Close Finder or preview windows that show it and run again.")
    out.mkdir(parents=True, exist_ok=True)

    manifest = build(settings, scenes, out)
    if not manifest["bags"]:
        raise SystemExit("No bag could be built with these settings.")
    (out / "bags.json").write_text(json.dumps(manifest, indent=2))
    (out / "config.json").write_text(json.dumps(settings, indent=2))
    write_html(manifest, out)
    print(summary(manifest))
    print(f"\nwritten to {out}\nopen {out / 'index.html'}")


if __name__ == "__main__":
    sys.exit(main())
