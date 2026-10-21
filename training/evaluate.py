"""Four metrics per method in one run.

From the bags in bags_eval (made by panorama/make_bags.py):
  AUROC                 how well the method ranks same-scene frame pairs above different-scene pairs, no threshold
  ARI                   how well the predicted groups match the true scenes (0 is chance, 1 is perfect)
  Perfect bag ratio     the share of bags grouped with no error at all
From the retail sequences themselves, each one given alone with all its frames:
  Cluster completeness  the share of a sequence's frames that stay together in one group, averaged over sequences

Per method one clustering threshold is used for everything: the one with the best mean ARI over the bags.
Methods are the DINOv3 poolings (dino/*, with -c for centered) and the first --layers blocks of VGGT-Omega.

    uv run python training/evaluate.py --bags-dir bags_eval --retail-dir RETAIL_DIR --device mps
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


def describe(paths: list[str], models: dict, args, device, context):
    """Affinity matrices of every method for one set of frames, and the DINO descriptors that centering needs."""
    images = torch.stack([baselines.load_image(p, args.height, args.width) for p in paths]).float() / 255
    on_device = images.to(device)
    found = {"dino": baselines.dino_features(models["plain"].aggregator.patch_embed, on_device, context)}
    if "dinov3" in models:
        found["dinov3"] = baselines.dino_features(models["dinov3"], on_device, context)
    affinities = {}
    for prefix, features in found.items():
        affinities.update(baselines.dino_affinities(features, prefix))
    wants_attention = not args.methods or any(m.startswith("attn") for m in args.methods)
    affinities.update(baselines.backbone_affinities(models["plain"], on_device, args.layers, context, wants_attention))
    if "trained" in models:
        affinities.update(baselines.learned_affinities(models["trained"], on_device))
    return affinities, found


def add_centered(items: list) -> None:
    """The centering mean is taken over every frame of the set, so it is one set of numbers for the whole run."""
    for prefix in items[0][1]:
        extra = baselines.centered_affinities([found[prefix] for _, found in items], prefix)
        for (affinities, _), more in zip(items, extra):
            affinities.update(more)


def bag_files(root: Path, bag: dict) -> list[str]:
    return [str(root / f["file"]) if (root / f["file"]).exists() else f["source"] for f in bag["frames"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bags-dir", default="bags_eval", help="folder made by panorama/make_bags.py")
    parser.add_argument("--retail-dir", help="scene folders of the retail data (default: $RETAIL_DIR, else ./RETAIL_DIR)")
    parser.add_argument("--weights", default="checkpoints/vggt_omega_1b_512.pt")
    parser.add_argument("--checkpoint", help="train.py checkpoint, adds the learned route and final heads")
    parser.add_argument("--dinov3", help="official DINOv3 ViT-L/16 .pth, adds dinov3/*")
    parser.add_argument("--route-layer", type=int, default=11)
    parser.add_argument("--layers", type=int, nargs="+", default=[4], help="1-based VGGT-Omega blocks")
    parser.add_argument(
        "--methods", nargs="+",
        help="only these, by name without the block number: dino/patch frame/patch global/patch (all by default)",
    )
    parser.add_argument("--limit", type=int, help="only the first N bags and N sequences, for a quick check")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="evaluation.json")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    bags_root = Path(args.bags_dir).expanduser()
    if not (bags_root / "bags.json").exists():
        raise SystemExit(f"{bags_root}/bags.json not found. Run panorama/make_bags.py first.")
    bags = json.loads((bags_root / "bags.json").read_text())["bags"]

    retail = Path(args.retail_dir or os.environ.get("RETAIL_DIR") or "RETAIL_DIR").expanduser()
    if not retail.is_dir():
        raise SystemExit(f"{retail} is not a directory. Pass --retail-dir.")
    scenes = baselines.read_scenes(retail)
    if args.limit:
        bags, scenes = bags[: args.limit], dict(list(scenes.items())[: args.limit])
    if not bags or not scenes:
        raise SystemExit("No bags or no sequences to evaluate.")

    device = baselines.pick_device(args.device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    models = {"plain": baselines.load_plain(args.weights, device)}
    if args.checkpoint:
        models["trained"] = baselines.load_trained(args.checkpoint, device, args.route_layer)
    if args.dinov3:
        models["dinov3"] = baselines.load_dinov3(args.dinov3, device)

    bag_items = [describe(bag_files(bags_root, bag), models, args, device, context)
                 for bag in tqdm(bags, desc="bags")]
    sequence_items = [describe(paths, models, args, device, context)
                      for paths in tqdm(scenes.values(), desc="sequences")]
    add_centered(bag_items)
    add_centered(sequence_items)

    labels = [torch.tensor([f["label"] for f in bag["frames"]]) for bag in bags]
    methods = sorted(set.intersection(*(set(a) for a, _ in bag_items + sequence_items)))
    if args.methods:
        methods = [m for m in methods if m.split("@")[0] in args.methods]
        if not methods:
            raise SystemExit(f"None of {args.methods} was computed. Names look like dino/patch or frame/patch.")
    rows = {}
    for method in tqdm(methods, desc="scoring", leave=False):
        affs = [a[method] for a, _ in bag_items]
        threshold = baselines.fit_threshold(affs, labels)
        pairs = [baselines.pair_scores(a, l) for a, l in zip(affs, labels)]
        ari, perfect = baselines.mean_ari(affs, labels, threshold)
        completeness = float(np.mean([baselines.largest_group_share(a[method], threshold) for a, _ in sequence_items]))
        rows[method] = {
            "auroc": baselines.auroc(torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])),
            "ari": ari,
            "perfect_bag_ratio": perfect,
            "cluster_completeness": completeness,
            "threshold": threshold,
        }

    frames = sum(len(bag["frames"]) for bag in bags)
    print(f"\n{len(bags)} bags ({frames} frames) for AUROC, ARI and perfect bags; "
          f"{len(scenes)} sequences ({sum(len(p) for p in scenes.values())} frames) for completeness.\n")
    header = f"{'method':<18}{'AUROC':>8}{'ARI':>8}{'perfect bags':>14}{'completeness':>14}{'threshold':>11}"
    print(header)
    print("-" * len(header))
    def order(item):
        base, _, layer = item[0].partition("@")
        known = baselines.VARIANTS.index(base) if base in baselines.VARIANTS else len(baselines.VARIANTS)
        return (known, int(layer or 0))

    ranked = sorted(rows.items(), key=order if args.methods else (lambda item: -item[1]["ari"]))
    for method, row in ranked:
        print(f"{method:<18}{row['auroc']:>8.3f}{row['ari']:>8.3f}{row['perfect_bag_ratio']:>14.1%}"
              f"{row['cluster_completeness']:>14.1%}{row['threshold']:>11.3f}")
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
