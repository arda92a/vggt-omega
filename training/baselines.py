"""Can the released VGGT-Omega already tell scenes apart? Baselines for splitting a bag of frames into scenes.

Random bags (2-4 scenes, 3-6 frames each) are drawn from the scene folders. For every bag each method
produces a frame-by-frame affinity matrix, which is clustered with the same average-linkage rule as the
model. Methods, at every layer in --layers (1-based):

  dino/cls        cosine of the DINOv3 class token of one frame, before any alternating attention.
  dino/patch      same with the mean of the patch tokens. These use the DINOv3 weights inside the released
                  checkpoint (fine-tuned). --dinov3 PATH adds dinov3/* with the original weights.
  dino/max        max over the patch tokens, per channel
  dino/gem        GeM pooling of the patch tokens, (mean x^p)^(1/p) with p = GEM_P, negatives clamped to 0
  dino/*-c        any of the above after subtracting the mean descriptor of all frames of all bags (centering),
                  which removes the part every frame shares
  frame/patch     cosine of the mean patch token after the frame-attention step of block L. Earlier blocks'
                  global steps are already mixed in, so only block 1 has seen each frame alone.
  frame/max, frame/gem   the same with max and GeM pooling of the patch tokens
  frame/reg       same with the mean of the 16 register tokens
  global/patch    cosine of the mean patch token after the global-attention step of block L (sees all frames)
  global/max, global/gem the same with max and GeM pooling
  global/reg      same with the register tokens
  cat/patch       mean patch token of the frame and global tokens joined, 2048 wide, as the heads read them
  attn            attention mass the frames of the bag give each other in the global-attention block
  */match-mean    patch matching, no pooling: every patch of frame A is compared with every patch of frame B
                  (cosine), each patch keeps its best match, and the score is the mean of those best matches
                  over the patches of A and of B. For dino, dinov3, frame and global patches.
  */matchNN       the same, but the score is the share of patches whose best match is at least 0.NN, for example
                  frame/match70@4: the share of patches of the two frames that find a match of cosine >= 0.70
  */match*-c      the same after subtracting the mean patch of the bag, which removes what every patch shares
  learned/*       the trained route and final heads, with --checkpoint

Nothing is trained and all scenes form one pool. Per method one threshold is chosen, the one with the best mean
ARI over all bags. Columns:
  auroc   threshold-free: do same-scene pairs score higher than different-scene pairs (0.5 is chance)
  ari     adjusted Rand index of the clustering at that threshold (0 is chance, 1 is perfect)
  exact   perfect bags: the share of bags clustered with no error at all

    uv run python training/baselines.py --data-dir MERGED_DIR --retail-dir RETAIL_DIR
"""

import argparse
import contextlib
import json
import math
import os
import random
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from vggt_omega.models import VGGTOmega
from vggt_omega.models.aggregator import slice_expand_and_flatten
from vggt_omega.models.multiscene import scene_groups
from vggt_omega.utils.scenes import adjusted_rand_index

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VARIANTS = [f"{prefix}/{kind}{suffix}" for prefix in ("dino", "dinov3") for suffix in ("", "-c")
            for kind in ("cls", "patch", "max", "gem")] + [
    "frame/patch", "frame/max", "frame/gem", "frame/reg", "global/patch", "global/max", "global/gem", "global/reg",
    "cat/patch", "attn", "attn-clean", "attn-peak", "attn-logit"
]
# Patch thresholds of the matchNN methods, around where best matches fall on bags_eval: raw cosines have a median
# of about 0.80 for same-scene frame pairs and 0.62 for different scenes, centered ones 0.67 and 0.42.
MATCH_THRESHOLDS = {"": (0.6, 0.7, 0.75, 0.8, 0.9), "-c": (0.4, 0.5, 0.6, 0.7, 0.8)}
VARIANTS += [f"{prefix}/match{kind}{suffix}" for prefix in ("dino", "dinov3", "frame", "global")
             for suffix in MATCH_THRESHOLDS
             for kind in ["-mean"] + [f"{round(100 * t)}" for t in MATCH_THRESHOLDS[suffix]]]
GEM_P = 3.0
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
LEARNED = ["learned/route", "learned/final"]


# ---------------------------------------------------------------- scenes and bags


def read_scenes(root: Path) -> dict[str, list[str]]:
    scenes = {}
    for folder in sorted(p for p in root.iterdir() if (p / "images").is_dir()):
        names = sorted(n for n in os.listdir(folder / "images") if n.lower().endswith(IMAGE_TYPES))
        if len(names) >= 3:
            scenes[folder.name] = [str(folder / "images" / n) for n in names]
    return scenes


def scenes_by_source(roots: list[Path]) -> dict[str, dict[str, list[str]]]:
    """{source: {scene: [paths]}}. A scene's source is the start of its folder name."""
    pools = defaultdict(dict)
    for root in roots:
        for name, paths in read_scenes(root).items():
            pools[name.split("_")[0]][name] = paths
    return pools


def sample_bag(pool: dict[str, list[str]], rng: random.Random, max_frames: int = 16):
    names = sorted(pool)
    count = min(rng.randint(2, 4), len(names), max_frames // 3)
    chosen = rng.sample(names, count)
    sizes = [rng.randint(3, min(6, len(pool[n]))) for n in chosen]
    while sum(sizes) > max_frames:
        sizes[sizes.index(max(sizes))] -= 1
    paths, labels = [], []
    for label, (name, size) in enumerate(zip(chosen, sizes)):
        for path in rng.sample(pool[name], size):
            paths.append(path)
            labels.append(label)
    order = list(range(len(paths)))
    rng.shuffle(order)
    return [paths[i] for i in order], torch.tensor([labels[i] for i in order])


@lru_cache(maxsize=6000)
def load_image(path: str, height: int, width: int) -> torch.Tensor:
    """Centre crop to the target aspect ratio, then resize. Every frame gets one shape, so padding cannot leak the source."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        if w / h > width / height:
            new_w = int(round(h * width / height))
            left = (w - new_w) // 2
            image = image.crop((left, 0, left + new_w, h))
        else:
            new_h = int(round(w * height / width))
            top = (h - new_h) // 2
            image = image.crop((0, top, w, top + new_h))
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


# ---------------------------------------------------------------- affinities


def cosine(features: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    return (features @ features.T).cpu()


def attention_affinities(block, tokens: torch.Tensor, patch_start: int, chunk: int = 512) -> dict:
    """How strongly the frames attend to each other in one global-attention step, four ways. All are symmetrised.

    attn         share of each query's attention that lands on every frame, over all keys, times N (uniform is 1)
    attn-clean   the same over the patch keys of the other frames only, so camera/register sinks and the own frame
                 do not take part, times N-1 (uniform is 1). Patch queries only.
    attn-peak    the largest attention weight a query puts on any single patch of the other frame, with the softmax
                 of attn-clean, times P(N-1). Sharp matches score high, diffuse attention scores about 1.
    attn-logit   log of the mean exp(q.k/sqrt(d)) over the patches of the other frame, no softmax across frames,
                 so it does not depend on how many frames are in the bag.
    """
    num_frames, per_frame, dim = tokens.shape
    patches = per_frame - patch_start
    attn, heads = block.attn, block.attn.num_heads
    x = block.norm1(tokens.reshape(1, num_frames * per_frame, dim))
    q, k, _ = attn.qkv(x).reshape(1, num_frames * per_frame, 3, heads, dim // heads).unbind(2)
    q, k = attn.q_norm(q.transpose(1, 2)), attn.k_norm(k.transpose(1, 2))
    k = k.float().transpose(-1, -2)
    device = tokens.device
    index = torch.arange(num_frames * per_frame, device=device)
    owner, is_patch = index // per_frame, (index % per_frame) >= patch_start
    frames = torch.arange(num_frames, device=device)
    sums = {name: torch.zeros(num_frames, num_frames, device=device) for name in ("attn", "attn-clean", "attn-peak", "attn-logit")}

    for start in range(0, num_frames * per_frame, chunk):
        end = min(start + chunk, num_frames * per_frame)
        logits = (q[0, :, start:end].float() @ k[0]) * attn.scale  # heads, queries, all keys
        rows = owner[start:end]
        sums["attn"].index_add_(0, rows, logits.softmax(-1).view(heads, end - start, num_frames, per_frame).sum(-1).mean(0))

        keep = is_patch[start:end]
        if not keep.any():
            continue
        on_patches = logits[:, keep].view(heads, int(keep.sum()), num_frames, per_frame)[..., patch_start:]
        rows = rows[keep]
        sums["attn-logit"].index_add_(0, rows, (on_patches.logsumexp(-1) - math.log(patches)).mean(0))
        own = frames[None, :] == rows[:, None]  # queries, frames
        others = on_patches.masked_fill(own[None, :, :, None], float("-inf"))
        probs = others.flatten(2).softmax(-1).view_as(others)
        sums["attn-clean"].index_add_(0, rows, probs.sum(-1).mean(0))
        sums["attn-peak"].index_add_(0, rows, probs.amax(-1).mean(0))

    scale = {
        "attn": num_frames / per_frame,
        "attn-clean": (num_frames - 1) / patches,
        "attn-peak": (num_frames - 1),
        "attn-logit": 1.0 / patches,
    }
    return {name: (0.5 * (m + m.T) * scale[name]).cpu() for name, m in sums.items()}


def pool_patches(patches: torch.Tensor) -> dict:
    """(N, P, C) patch tokens to one vector per frame: mean, max and GeM (negatives clamped to 0 before the power)."""
    return {
        "patch": patches.mean(1),
        "max": patches.max(1).values,
        "gem": patches.clamp(min=1e-6).pow(GEM_P).mean(1).pow(1.0 / GEM_P),
    }


@torch.no_grad()
def dino_features(backbone, images: torch.Tensor, context) -> dict:
    """Descriptors of a DINOv3 backbone that sees each frame alone: class token and three poolings of the patches."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    with context:
        out = backbone((images - mean) / std)
    pooled = {kind: vectors.cpu() for kind, vectors in pool_patches(out["x_norm_patchtokens"].float()).items()}
    return {"cls": out["x_norm_clstoken"].float().cpu(), **pooled}


def dino_affinities(features: dict, name: str) -> dict:
    return {f"{name}/{kind}": cosine(vectors) for kind, vectors in features.items()}


def best_matches(patches: torch.Tensor) -> torch.Tensor:
    """patches (N, P, C), unit length -> best (N, N, P): best[i, j, p] = max over q of cos(patch p of i, patch q of j)."""
    best = torch.empty(patches.shape[0], patches.shape[0], patches.shape[1], device=patches.device)
    for i, frame in enumerate(patches):
        best[i] = torch.einsum("pc,jqc->jpq", frame, patches).amax(-1)
    return best


@torch.no_grad()
def match_affinities(patches: torch.Tensor, name: str) -> dict:
    """Patch-matching affinities of one set of frames from all their patch tokens (N, P, C), raw and centered."""
    affinities = {}
    for suffix, thresholds in MATCH_THRESHOLDS.items():
        x = patches.float()
        if suffix == "-c":
            x = x - x.mean((0, 1), keepdim=True)  # mean patch of the whole bag, no labels involved
        best = best_matches(F.normalize(x, dim=-1))

        def symmetric(scores: torch.Tensor) -> torch.Tensor:  # (N, N): mean of the A->B and B->A scores
            return (0.5 * (scores + scores.T)).cpu()

        affinities[f"{name}/match-mean{suffix}"] = symmetric(best.mean(-1))
        for threshold in thresholds:
            affinities[f"{name}/match{round(100 * threshold)}{suffix}"] = symmetric((best >= threshold).float().mean(-1))
    return affinities


@torch.no_grad()
def dino_patches(backbone, images: torch.Tensor, context) -> torch.Tensor:
    """Patch tokens (N, P, C) of a DINOv3 backbone that sees each frame alone."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    with context:
        return backbone((images - mean) / std)["x_norm_patchtokens"].float()


def centered_affinities(items: list[dict], name: str) -> list[dict]:
    """Subtract each descriptor's mean over every frame of every item, then take the cosine."""
    means = {kind: torch.cat([item[kind] for item in items]).mean(0) for kind in items[0]}
    return [{f"{name}/{kind}-c": cosine(item[kind] - means[kind]) for kind in item} for item in items]


@torch.no_grad()
def backbone_affinities(
    model, images: torch.Tensor, layers: list[int], context, attention: bool = True, match: bool = False
) -> dict:
    """Affinity matrices of every backbone method at the given 1-based blocks. attention=False skips the costly attn,
    match=True adds the patch-matching methods (frame/match*, global/match*)."""
    aggregator = model.aggregator
    num_frames, _, height, width = images.shape
    affinities = {}
    with context:
        x = ((images[None] - aggregator._resnet_mean) / aggregator._resnet_std).view(num_frames, 3, height, width)
        camera = slice_expand_and_flatten(aggregator.camera_token, 1, num_frames)
        register = slice_expand_and_flatten(aggregator.register_token, 1, num_frames)
        patch = aggregator.patch_embed(x)
        if isinstance(patch, dict):
            patch = patch["x_norm_patchtokens"]
        tokens = torch.cat([camera, register, patch], dim=1)
        _, per_frame, dim = tokens.shape
        sin, cos = aggregator.rope_embed(H=height // aggregator.patch_size, W=width // aggregator.patch_size)
        rope = (sin.to(images.device, torch.float32), cos.to(images.device, torch.float32))
        start = aggregator.patch_token_start

        for index in range(max(layers)):
            tokens, frame_tokens = aggregator._run_frame_block(tokens, 1, num_frames, per_frame, dim, index, rope)
            kind = aggregator.inter_frame_attention_types[index]
            layer = index + 1
            if attention and layer in layers and kind == "global":
                for name, matrix in attention_affinities(aggregator.inter_frame_blocks[index], tokens, start).items():
                    affinities[f"{name}@{layer}"] = matrix
            tokens = aggregator._run_inter_frame_attention_block(
                tokens, 1, num_frames, per_frame, dim, index, kind, None
            )
            if layer in layers:
                after = tokens.reshape(num_frames, per_frame, dim)
                for stream, seen in (("frame", frame_tokens[0]), ("global", after)):
                    for kind, vectors in pool_patches(seen[:, start:].float()).items():
                        affinities[f"{stream}/{kind}@{layer}"] = cosine(vectors)
                    affinities[f"{stream}/reg@{layer}"] = cosine(seen[:, 1:start].float().mean(1))
                    if match:
                        for method, matrix in match_affinities(seen[:, start:], stream).items():
                            affinities[f"{method}@{layer}"] = matrix
                # The heads read frame and global tokens side by side, 2 x 1024 wide.
                joined = torch.cat([frame_tokens[0], after], dim=-1).float()
                affinities[f"cat/patch@{layer}"] = cosine(joined[:, start:].mean(1))
    return affinities


@torch.no_grad()
def learned_affinities(model, images: torch.Tensor) -> dict:
    prediction = model(images[None])
    return {
        "learned/route": prediction["route_affinity"][0].float().cpu(),
        "learned/final": prediction["affinity"][0].float().cpu(),
    }


# ---------------------------------------------------------------- scoring


def pair_scores(affinity: torch.Tensor, labels: torch.Tensor):
    first, second = torch.triu_indices(len(labels), len(labels), 1)
    return affinity[first, second], labels[first] == labels[second]


def auroc(scores: torch.Tensor, same: torch.Tensor) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    values, inverse, counts = scores.unique(return_inverse=True, return_counts=True)
    ranks = (counts.cumsum(0) - (counts - 1) / 2)[inverse]
    positive = int(same.sum())
    negative = len(same) - positive
    return float((ranks[same].sum() - positive * (positive + 1) / 2) / (positive * negative))


def mean_ari(affinities: list, labels: list, threshold: float, linkage: str = "average") -> tuple[float, float]:
    aris, exact = [], []
    for affinity, label in zip(affinities, labels):
        groups = scene_groups(affinity[None], threshold, linkage)[0]
        aris.append(adjusted_rand_index(groups, label))
        same = groups[:, None] == groups[None, :]
        exact.append(float((same == (label[:, None] == label[None, :])).all()))
    return float(np.mean(aris)), float(np.mean(exact))


def threshold_grid(affinities: list, labels: list, points: int = 12) -> list[float]:
    scores = torch.cat([pair_scores(a, l)[0] for a, l in zip(affinities, labels)])
    return torch.quantile(scores, torch.linspace(0.2, 0.95, points)).tolist()


def fit_threshold(affinities: list, labels: list, linkage: str = "average") -> float:
    grid = threshold_grid(affinities, labels)
    return max(grid, key=lambda threshold: mean_ari(affinities, labels, threshold, linkage)[0])


def largest_group_share(affinity: torch.Tensor, threshold: float, linkage: str = "average") -> float:
    """Cluster completeness of one scene given alone: the share of its frames that stay in one group."""
    groups = scene_groups(affinity[None], threshold, linkage)[0]
    return float(torch.bincount(groups).max()) / len(groups)


class Bag:
    def __init__(self, labels, affinities, descriptors):
        self.labels, self.affinities, self.descriptors = labels, affinities, descriptors


def collect_bags(pool, count, seed, args, models, device, context) -> list:
    rng = random.Random(seed)
    bags = []
    for _ in tqdm(range(count), leave=False):
        paths, labels = sample_bag(pool, rng)
        images = torch.stack([load_image(p, args.height, args.width) for p in paths]).float() / 255
        descriptors = {"dino": dino_features(models["plain"].aggregator.patch_embed, images.to(device), context)}
        if "dinov3" in models:
            descriptors["dinov3"] = dino_features(models["dinov3"], images.to(device), context)
        affinities = {}
        for name, features in descriptors.items():
            affinities.update(dino_affinities(features, name))
        backbone = backbone_affinities(models["plain"], images.to(device), args.layers, context)
        affinities.update(backbone)
        if "trained" in models:
            affinities.update(learned_affinities(models["trained"], images.to(device)))
        bags.append(Bag(labels, affinities, descriptors))
    return bags


def add_centered(bags_by_condition: dict) -> None:
    """The centering mean comes from every frame of every bag, so it is one number set for the whole run."""
    everything = [bag for bags in bags_by_condition.values() for bag in bags]
    for name in everything[0].descriptors:
        for bag, extra in zip(everything, centered_affinities([b.descriptors[name] for b in everything], name)):
            bag.affinities.update(extra)


def evaluate(bags_by_condition: dict) -> dict:
    """results[condition][method] = scores. One threshold per method: the best mean ARI over all bags."""
    everything = [bag for bags in bags_by_condition.values() for bag in bags]
    methods = sorted(set.intersection(*(set(b.affinities) for b in everything)))
    results = {condition: {} for condition in bags_by_condition}
    thresholds = {}
    for method in tqdm(methods, desc="scoring", leave=False):
        thresholds[method] = fit_threshold([b.affinities[method] for b in everything], [b.labels for b in everything])
        for condition, bags in bags_by_condition.items():
            affs = [b.affinities[method] for b in bags]
            labels = [b.labels for b in bags]
            pairs = [pair_scores(a, l) for a, l in zip(affs, labels)]
            ari, exact = mean_ari(affs, labels, thresholds[method])
            results[condition][method] = {
                "auroc": auroc(torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])),
                "ari": ari,
                "exact": exact,
            }
    return {"results": results, "thresholds": thresholds}


# ---------------------------------------------------------------- output


def print_tables(results: dict, layers: list[int], counts: dict) -> None:
    variants = VARIANTS + LEARNED
    for condition, methods in results.items():
        print(f"\n==== {condition}: {counts[condition]} bags ====")
        for key, title in (("auroc", "AUROC"), ("ari", "ARI"), ("exact", "Perfect bag ratio")):
            print(f"\n  {title:<28}" + "".join(f"L{layer:<6}" for layer in layers) + "-")
            for variant in variants:
                cells = [methods.get(f"{variant}@{layer}") for layer in layers] + [methods.get(variant)]
                if not any(cells):
                    continue
                print(f"  {variant:<28}" + "".join(f"{c[key]:<7.3f}" if c else "-      " for c in cells))
        best = max(methods, key=lambda m: methods[m]["ari"])
        print(f"\nbest ARI: {best} {methods[best]['ari']:.3f} (perfect bags {methods[best]['exact']:.0%}, "
              f"AUROC {methods[best]['auroc']:.3f})")


# ---------------------------------------------------------------- models


def load_plain(weights: str, device: torch.device) -> VGGTOmega:
    state = torch.load(weights, map_location="cpu", weights_only=False)
    state = state.get("model", state) if isinstance(state, dict) else state
    model = VGGTOmega(enable_camera=False, enable_depth=False)
    result = model.load_state_dict(state, strict=False)
    if any(key.startswith("aggregator.") for key in result.missing_keys):
        raise SystemExit(f"{weights} is missing aggregator weights")
    return model.to(device).eval()


def load_dinov3(path: str, device: torch.device):
    """An official bare DINOv3 ViT/16 state dict. The size (S, B or L) is read from the weights.

    ViT-L is built the way checkpoint.dinov3_weight_path is (as before). ViT-S and ViT-B use the settings of the
    official dinov3/hub/backbones.py.
    """
    from vggt_omega.models.aggregator import _build_patch_embed
    from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer

    state = torch.load(path, map_location="cpu", weights_only=False)
    state = {key: value for key, value in state.items() if "local_cls_norm" not in key}
    width = state["cls_token"].shape[-1]
    if width == 1024:
        model = _build_patch_embed(patch_size=16, embed_dim=1024)
    elif width in (384, 768):
        model = DinoVisionTransformer(
            img_size=224, patch_size=16, pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_dtype="fp32", embed_dim=width, depth=12, num_heads=width // 64, ffn_ratio=4,
            qkv_bias=True, layerscale_init=1.0e-5, norm_layer="layernormbf16", ffn_layer="mlp", ffn_bias=True,
            proj_bias=True, n_storage_tokens=4, mask_k_bias=True,
        )
    else:
        raise SystemExit(f"{path}: width {width} is not a DINOv3 ViT-S/16, ViT-B/16 or ViT-L/16")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_trained(checkpoint: str, device: torch.device, route_layer: int) -> VGGTOmega:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded.get("model", loaded)
    if not any(key.startswith("multiscene.") for key in state):
        raise SystemExit(f"{checkpoint} has no multiscene.* weights. Use a checkpoint from train.py.")
    model = VGGTOmega(
        embed_dim=state["aggregator.camera_token"].shape[-1],
        enable_camera=False,
        enable_depth=False,
        autocast=device.type == "cuda",
        multiscene={
            "enabled": True,
            "route_layer": route_layer,
            "patch_pooling": "attention" if "multiscene.affinity.pool.queries" in state
            else "mean" if "multiscene.affinity.patch_proj.weight" in state else "none",
            "num_queries": state["multiscene.affinity.pool.queries"].shape[0]
            if "multiscene.affinity.pool.queries" in state else 8,
        },
    )
    result = model.load_state_dict(state, strict=False)
    if any(key.startswith(("aggregator.", "multiscene.")) for key in result.missing_keys):
        raise SystemExit(f"{checkpoint} is missing aggregator or multiscene weights")
    return model.to(device).eval()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", help="merged dataset root (default: $MERGED_DIR, else ./MERGED_DIR)")
    parser.add_argument("--retail-dir", help="images-only dataset root (default: $RETAIL_DIR, else ./RETAIL_DIR if it exists)")
    parser.add_argument("--weights", default="checkpoints/vggt_omega_1b_512.pt", help="released VGGT-Omega weights")
    parser.add_argument("--checkpoint", help="train.py checkpoint, adds the learned route and final heads")
    parser.add_argument("--dinov3", help="official DINOv3 ViT-L/16 .pth, adds dinov3/cls and dinov3/patch")
    parser.add_argument("--route-layer", type=int, default=11, help="route_layer the checkpoint was trained with")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24], help="1-based block numbers")
    parser.add_argument("--bags", type=int, default=100, help="random bags per condition")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="baselines.json")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    roots = []
    for explicit, env, default in ((args.data_dir, "MERGED_DIR", "MERGED_DIR"), (args.retail_dir, "RETAIL_DIR", "RETAIL_DIR")):
        path = Path(explicit or os.environ.get(env) or default).expanduser()
        if path.is_dir():
            roots.append(path)
        elif explicit:
            raise SystemExit(f"{explicit} is not a directory")
    if not roots:
        raise SystemExit("No dataset folder found. Pass --data-dir and --retail-dir.")

    pools = scenes_by_source(roots)
    print("scenes: " + ", ".join(f"{source} {len(scenes)}" for source, scenes in sorted(pools.items())))

    conditions = {source: scenes for source, scenes in pools.items() if len(scenes) >= 2}
    everything = {name: paths for scenes in pools.values() for name, paths in scenes.items()}
    if len(pools) >= 2:
        conditions["mixed"] = everything
    if not conditions:
        raise SystemExit("Need at least two scenes in one source.")

    device = pick_device(args.device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    models = {"plain": load_plain(args.weights, device)}
    if args.checkpoint:
        models["trained"] = load_trained(args.checkpoint, device, args.route_layer)
    if args.dinov3:
        models["dinov3"] = load_dinov3(args.dinov3, device)

    bags = {}
    for condition, pool in conditions.items():
        print(f"bags: {condition}")
        bags[condition] = collect_bags(pool, args.bags, 0, args, models, device, context)

    add_centered(bags)
    scored = evaluate(bags)
    print_tables(scored["results"], args.layers, {c: len(b) for c, b in bags.items()})
    Path(args.out).write_text(json.dumps(scored, indent=2))
    print(f"\nfull results written to {args.out}")


if __name__ == "__main__":
    main()
