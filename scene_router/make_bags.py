"""Build train/val bags from Stitch_Requests for the scene router.

A bag is a shuffled mix of frames from 2-4 different stitch requests (sequences), 10-15 frames in total. The router
learns to tell which frames of a bag come from the same sequence.

How bags are made
  1. Split: sequences are split into train and val by audit_id, so the shelves of one store visit never end up in
     both sets.
  2. Rounds: in one round every frame of every sequence goes into exactly one bag. A 60-frame sequence is cut into
     runs of neighbouring frames that land in ~7-8 different bags, a 4-frame sequence lands in one bag.
     Each round cuts the sequences differently, so more rounds give more (and different) bags from the same frames.
  3. Each bag picks its size (10-15) and its number of sequences (2-4) at random, splits the size into one share
     per sequence, then takes a run of that many frames from sequences that still have frames left in this round.
     Sequences with more frames left are picked more often, so long sequences spread over many bags.

Output (in --out):
  splits.json   which sequences are train and which are val
  train.json    the train bags (--train-rounds rounds)
  val.json      the val bags (--val-rounds rounds, fixed seed, so every run validates on the same bags)

    uv run python scene_router/make_bags.py
    uv run python scene_router/make_bags.py --train-rounds 30 --val-fraction 0.15 --out scene_router/data
"""

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class Sequence:
    name: str  # folder name in Stitch_Requests
    audit_id: int
    frames: list[str]  # paths relative to the data root, in capture order


# ---------------------------------------------------------------- reading and splitting


def read_sequences(root: Path) -> list[Sequence]:
    sequences = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        request = folder / "request.json"
        frames = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_TYPES)
        if not request.exists() or not frames:
            continue
        audit_id = json.loads(request.read_text())["audit_id"]
        sequences.append(Sequence(folder.name, audit_id, [f"{folder.name}/{f}" for f in frames]))
    return sequences


def split_by_audit(sequences: list[Sequence], val_fraction: float, seed: int) -> tuple[list[Sequence], list[Sequence]]:
    """Whole audits go to val until it holds about val_fraction of the frames."""
    by_audit: dict[int, list[Sequence]] = {}
    for sequence in sequences:
        by_audit.setdefault(sequence.audit_id, []).append(sequence)
    audits = sorted(by_audit)
    random.Random(seed).shuffle(audits)

    total = sum(len(s.frames) for s in sequences)
    val, train, val_frames = [], [], 0
    for audit in audits:
        if val_frames < val_fraction * total:
            val += by_audit[audit]
            val_frames += sum(len(s.frames) for s in by_audit[audit])
        else:
            train += by_audit[audit]
    return train, val


# ---------------------------------------------------------------- bags


def split_size(size: int, parts: int, min_part: int, rng: random.Random) -> list[int]:
    """Random split of `size` into `parts` shares, each at least min_part."""
    shares = [min_part] * parts
    for _ in range(size - min_part * parts):
        shares[rng.randrange(parts)] += 1
    return shares


def make_round(sequences: list[Sequence], args, rng: random.Random) -> list[list[tuple[str, str]]]:
    """One round: every frame of every sequence in exactly one bag. A bag is a list of (frame path, sequence name)."""
    # Frames left per sequence, as a queue in capture order (randomly reversed, randomly rotated start),
    # so each round cuts the sequences into different runs of neighbouring frames.
    left: dict[str, list[str]] = {}
    audit: dict[str, int] = {}
    for sequence in sequences:
        frames = list(sequence.frames)
        if rng.random() < 0.5:
            frames.reverse()
        start = rng.randrange(len(frames))
        left[sequence.name] = frames[start:] + frames[:start]
        audit[sequence.name] = sequence.audit_id

    bags = []
    while True:
        alive = [name for name, frames in left.items() if frames]
        if len(alive) < args.min_scenes:
            break
        size = rng.randint(args.min_frames, args.max_frames)
        scenes = rng.randint(args.min_scenes, min(args.max_scenes, len(alive)))
        scenes = min(scenes, size // args.min_run)

        # Pick the sequences, weighted by frames left, at most one per audit (two requests of one visit may show
        # the same shelf, which would make their "different scene" label wrong).
        picked, used_audits, candidates = [], set(), list(alive)
        while len(picked) < scenes and candidates:
            weights = [len(left[name]) for name in candidates]
            name = rng.choices(candidates, weights)[0]
            candidates.remove(name)
            if args.one_per_audit and audit[name] in used_audits:
                continue
            picked.append(name)
            used_audits.add(audit[name])
        if len(picked) < args.min_scenes:
            break

        takes = {name: 0 for name in picked}
        for name, share in zip(picked, split_size(size, len(picked), args.min_run, rng)):
            takes[name] = min(share, len(left[name]))
        # Sequences short on frames leave a gap: fill it from the picked sequences that still have frames,
        # then from new sequences while the bag has room for one more.
        fillers = [n for n in picked if len(left[n]) > takes[n]]
        others = [n for n in alive if n not in takes and not (args.one_per_audit and audit[n] in used_audits)]
        rng.shuffle(others)
        while sum(takes.values()) < size:
            if not fillers and others and len(takes) < args.max_scenes:
                name = others.pop()
                takes[name] = 0
                used_audits.add(audit[name])
                fillers.append(name)
            if not fillers:
                break
            name = rng.choice(fillers)
            takes[name] += 1
            if takes[name] == len(left[name]):
                fillers.remove(name)
        # A tail shorter than min_run is swallowed when the bag has room, so it is not left as a lone frame.
        for name in takes:
            tail = len(left[name]) - takes[name]
            if 0 < tail < args.min_run and sum(takes.values()) + tail <= args.max_frames:
                takes[name] += tail

        bag = []
        for name, take in takes.items():
            bag += [(frame, name) for frame in left[name][:take]]
            left[name] = left[name][take:]
        if len(bag) >= args.min_frames:  # only the last bags of a round can come out short: those are dropped
            rng.shuffle(bag)
            bags.append(bag)
    return bags


def make_bags(sequences: list[Sequence], rounds: int, args, seed: int, split: str) -> list[dict]:
    rng = random.Random(seed)
    bags = []
    for round_index in range(rounds):
        for bag in make_round(sequences, args, rng):
            names = list(dict.fromkeys(name for _, name in bag))  # sequences in order of first appearance
            bags.append({
                "id": f"{split}_r{round_index:02d}_b{len(bags):05d}",
                "frames": [frame for frame, _ in bag],
                "labels": [names.index(name) for _, name in bag],
                "sequences": names,
            })
    return bags


# ---------------------------------------------------------------- report


def describe(name: str, sequences: list[Sequence], bags: list[dict]) -> None:
    frames = sum(len(s.frames) for s in sequences)
    used = Counter(frame for bag in bags for frame in bag["frames"])
    sizes = Counter(len(bag["frames"]) for bag in bags)
    scenes = Counter(len(bag["sequences"]) for bag in bags)
    print(f"{name}: {len(sequences)} sequences, {frames} frames, {len(bags)} bags")
    print(f"  frames used at least once: {len(used)}/{frames}  ({100 * len(used) / max(frames, 1):.1f}%)")
    print(f"  bag sizes:       {dict(sorted(sizes.items()))}")
    print(f"  sequences / bag: {dict(sorted(scenes.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="Stitch_Requests", help="folder of stitch request folders")
    parser.add_argument("--out", default="scene_router/data")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="share of frames for val, split by audit")
    parser.add_argument("--train-rounds", type=int, default=20, help="each round uses every train frame once")
    parser.add_argument("--val-rounds", type=int, default=2)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--min-scenes", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, default=4)
    parser.add_argument("--min-run", type=int, default=2, help="fewest frames a sequence brings to a bag")
    parser.add_argument("--allow-same-audit", dest="one_per_audit", action="store_false",
                        help="let two requests of one audit share a bag (their labels may be wrong)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sequences = read_sequences(root)
    train, val = split_by_audit(sequences, args.val_fraction, args.seed)
    train_bags = make_bags(train, args.train_rounds, args, seed=args.seed + 1, split="train")
    val_bags = make_bags(val, args.val_rounds, args, seed=12345, split="val")

    meta = {"root": str(root.resolve()), "settings": vars(args)}
    (out / "splits.json").write_text(json.dumps({
        **meta,
        "train": [s.name for s in train],
        "val": [s.name for s in val],
    }, indent=1))
    for split, bags in (("train", train_bags), ("val", val_bags)):
        (out / f"{split}.json").write_text(json.dumps({**meta, "bags": bags}))

    describe("train", train, train_bags)
    describe("val", val, val_bags)
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
