"""Cluster completeness: every sequence is given alone, with all its frames, as a one-scene bag.

The right answer is a single group. The metric is the share of a sequence's frames that stay together in its largest
group, averaged over the sequences, at the threshold fitted by training/baselines.py.

    uv run python training/single_scene_test.py --retail-dir RETAIL_DIR --device mps \
        --height 448 --width 256 --thresholds baselines.json

--thresholds takes the baselines.py result files that hold the fitted thresholds ("thresholds" entry).
training/evaluate.py runs this together with the three bag metrics in one go.
"""

import argparse
import contextlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import baselines
from vggt_omega.models.multiscene import scene_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", help="merged dataset root (default: $MERGED_DIR, else ./MERGED_DIR if it exists)")
    parser.add_argument("--retail-dir", help="images-only dataset root (default: $RETAIL_DIR, else ./RETAIL_DIR if it exists)")
    parser.add_argument("--weights", default="checkpoints/vggt_omega_1b_512.pt")
    parser.add_argument("--checkpoint", help="train.py checkpoint, adds the learned route and final heads")
    parser.add_argument("--dinov3", help="official DINOv3 ViT-L/16 .pth, adds dinov3/cls and dinov3/patch")
    parser.add_argument("--route-layer", type=int, default=11)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24], help="1-based block numbers")
    parser.add_argument("--thresholds", nargs="*", default=["baselines.json"], help="baselines.py result files")
    parser.add_argument("--max-frames", type=int, default=24, help="skip sequences with more frames than this")
    parser.add_argument("--limit", type=int, help="only the first N sequences, for a quick check")
    parser.add_argument("--inspect", default="dino/patch", help="method whose split sequences are listed")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="single_scene.json")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    scenes = {}
    for explicit, env, default in ((args.data_dir, "MERGED_DIR", "MERGED_DIR"), (args.retail_dir, "RETAIL_DIR", "RETAIL_DIR")):
        path = Path(explicit or os.environ.get(env) or default).expanduser()
        if path.is_dir():
            scenes.update(baselines.read_scenes(path))
        elif explicit:
            raise SystemExit(f"{explicit} is not a directory")
    scenes = {name: paths for name, paths in scenes.items() if len(paths) <= args.max_frames}
    if args.limit:
        scenes = dict(list(scenes.items())[: args.limit])
    if not scenes:
        raise SystemExit("No sequences found. Pass --retail-dir and/or --data-dir.")

    fitted = {}
    for path in args.thresholds:
        if Path(path).exists():
            fitted.update(json.loads(Path(path).read_text()).get("thresholds", {}))
        else:
            print(f"no threshold file {path}")

    device = baselines.pick_device(args.device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    plain = baselines.load_plain(args.weights, device)
    trained = baselines.load_trained(args.checkpoint, device, args.route_layer) if args.checkpoint else None
    dinov3 = baselines.load_dinov3(args.dinov3, device) if args.dinov3 else None

    matrices: dict[str, dict[str, torch.Tensor]] = {}
    descriptors: dict[str, dict[str, dict]] = {}
    for name, paths in tqdm(scenes.items(), desc="sequences"):
        images = torch.stack([baselines.load_image(p, args.height, args.width) for p in paths]).float() / 255
        on_device = images.to(device)
        found = {"dino": baselines.dino_features(plain.aggregator.patch_embed, on_device, context)}
        if dinov3 is not None:
            found["dinov3"] = baselines.dino_features(dinov3, on_device, context)
        affinities = {}
        for prefix, features in found.items():
            descriptors.setdefault(prefix, {})[name] = features
            affinities.update(baselines.dino_affinities(features, prefix))
        affinities.update(baselines.backbone_affinities(plain, on_device, args.layers, context))
        if trained is not None:
            affinities.update(baselines.learned_affinities(trained, on_device))
        for method, affinity in affinities.items():
            matrices.setdefault(method, {})[name] = affinity

    for prefix, by_name in descriptors.items():
        for name, extra in zip(by_name, baselines.centered_affinities(list(by_name.values()), prefix)):
            for method, affinity in extra.items():
                matrices.setdefault(method, {})[name] = affinity
    names = list(scenes)
    sizes = [len(scenes[n]) for n in names]
    print(f"\n{len(names)} sequences, {min(sizes)}-{max(sizes)} frames each. The right answer is one group per sequence.\n")
    header = f"{'method':<18}{'threshold':>10}{'completeness':>14}"
    print(header)
    print("-" * len(header))
    rows, split = {}, {}
    for method in sorted(matrices):
        if method not in fitted:
            print(f"{method:<18}{'-':>10}{'-':>14}")
            continue
        tau = fitted[method]
        groups = [scene_groups(matrices[method][n][None], tau)[0] for n in names]
        share = float(np.mean([float(torch.bincount(g).max()) / len(g) for g in groups]))
        rows[method] = {"threshold": tau, "cluster_completeness": share}
        split[method] = [(n, int(g.unique().numel()), sorted(torch.bincount(g).tolist(), reverse=True))
                         for n, g in zip(names, groups) if g.unique().numel() > 1]
        print(f"{method:<18}{tau:>10.3f}{share:>14.1%}")

    if args.inspect in split:
        broken = sorted(split[args.inspect], key=lambda item: -item[1])
        print(f"\n{args.inspect}: {len(broken)} of {len(names)} sequences were split. Most groups first:")
        for name, count, group_sizes in broken[:20]:
            print(f"  {name:<22}{count} groups, sizes {group_sizes}")
    elif args.inspect:
        print(f"\nno fitted threshold for {args.inspect}, so no sequences to list (pass --thresholds)")

    Path(args.out).write_text(json.dumps({"methods": rows, "split": split}, indent=2))
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
