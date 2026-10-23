"""Local Gradio demo: upload frames from any scenes and see how VGGT-Omega groups them.

    uv run --extra ui python demo_multiscene.py --checkpoint checkpoints/vggt_omega_1b_512.pt

Methods:
  Patch matching   no training, best on retail: two frames are one scene when enough of their patches match
  Patch cosine     no training: cosine of the mean patch token of each frame
  Learned heads    only with a checkpoint that has multi-scene weights
"""

import argparse
import base64
import html
import io
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from vggt_omega.models import VGGTOmega
from vggt_omega.models.multiscene import scene_groups
from vggt_omega.utils import baseline
from vggt_omega.utils.baseline import patch_feature_affinity
from vggt_omega.utils.load_fn import load_and_preprocess_images

LEARNED = "Learned heads"
BASELINE = "Baseline: patch cosine (no training)"
MATCHING = "Patch matching (no training)"

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#16a5a5", "#d4a017", "#f032e6", "#7f7f00", "#808080"]
IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str, device: torch.device, route_layer: int, threshold: float, linkage: str) -> VGGTOmega:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
    del loaded
    if not any(key.startswith("multiscene.") for key in state):
        print(f"{checkpoint} has no multiscene.* weights: only the baseline method is available.")
        model = VGGTOmega(enable_camera=False, enable_depth=False)
        result = model.load_state_dict(state, strict=False)
        if any(key.startswith("aggregator.") for key in result.missing_keys):
            raise SystemExit(f"{checkpoint} is missing aggregator weights")
        return model.to(device).eval()
    model = VGGTOmega(
        embed_dim=state["aggregator.camera_token"].shape[-1],
        autocast=device.type == "cuda",
        multiscene={
            "enabled": True,
            "num_heads": 16,
            "threshold": threshold,
            "linkage": linkage,
            "route_layer": route_layer,
            "patch_pooling": "attention" if "multiscene.affinity.pool.queries" in state
            else "mean" if "multiscene.affinity.patch_proj.weight" in state else "none",
            "num_queries": state["multiscene.affinity.pool.queries"].shape[0]
            if "multiscene.affinity.pool.queries" in state else 8,
            "isolate_backbone": True,
            "isolate_camera": True,
        },
    )
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def group_frames(
    model: VGGTOmega, paths: list[str], resolution: int, threshold: float, linkage: str,
    method: str = LEARNED, baseline_layer: int = 4, baseline_threshold: float = 0.9,
    match_patch_threshold: float = baseline.MATCH_PATCH_THRESHOLD,
    match_threshold: float = baseline.MATCH_GROUP_THRESHOLD, match_centered: bool = True,
) -> dict:
    """Group frames. `maps` are the (title, matrix in [0, 1]) heatmaps to draw, `group_id` the predicted scenes."""
    device = next(model.parameters()).device
    if method == MATCHING:
        # Frames are prepared exactly as in the evaluation (448 x 256 centre crop), so the thresholds carry over.
        images = baseline.load_frames(paths).to(device)
        share = baseline.patch_match_affinity(
            model.aggregator, images, baseline_layer, float(match_patch_threshold), bool(match_centered)
        )
        groups = scene_groups(share[None], float(match_threshold), linkage)[0]
        top = max(float(share[~torch.eye(len(share), dtype=torch.bool)].max()), 1e-6)
        title = f"matched patch share, block {baseline_layer} (0 to {top:.2f})"
        return {"group_id": groups, "maps": [(title, (share / top).clamp(0, 1).numpy())]}
    images = load_and_preprocess_images(paths, mode="balanced", image_resolution=resolution).to(device)
    if method == BASELINE:
        affinity = patch_feature_affinity(model.aggregator, images, baseline_layer)
        groups = scene_groups(affinity[None], float(baseline_threshold), linkage)[0]
        off_diagonal = affinity[~torch.eye(len(affinity), dtype=torch.bool)]
        low, high = float(off_diagonal.min()), float(off_diagonal.max())
        stretched = ((affinity - low) / max(high - low, 1e-6)).clamp(0, 1)
        return {"group_id": groups, "maps": [(f"patch cosine, layer {baseline_layer} (stretched {low:.2f}-{high:.2f})", stretched.numpy())]}
    if model.multiscene is None:
        raise gr.Error("This checkpoint has no multi-scene weights. Use the baseline method.")
    model.multiscene.threshold = float(threshold)
    model.multiscene.linkage = linkage
    prediction = model(images)
    result = {key: prediction[key][0].float().cpu() for key in ("affinity", "route_affinity", "group_id")}
    result["maps"] = [("final affinity", result["affinity"].numpy()), ("route affinity", result["route_affinity"].numpy())]
    return result


# ---------------------------------------------------------------- views


def _thumbnail(path: str, width: int = 150) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((width, width * 2))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def scenes_html(paths: list[str], groups: torch.Tensor) -> str:
    """One card per predicted scene, with its frames."""
    cards = []
    for number, group in enumerate(sorted(groups.unique().tolist())):
        members = [i for i in range(len(paths)) if int(groups[i]) == group]
        color = PALETTE[number % len(PALETTE)]
        frames = "".join(
            f'<div style="width:156px;font-size:11px;word-break:break-all;text-align:center">'
            f'<img src="{_thumbnail(paths[i])}" style="width:150px;border-radius:4px"><br>{html.escape(Path(paths[i]).name)}</div>'
            for i in members
        )
        cards.append(
            f'<div style="border:2px solid {color};border-radius:8px;padding:8px;margin:8px 0">'
            f'<div style="font-weight:600;color:{color};margin-bottom:6px">Scene {number + 1} &middot; {len(members)} frames</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px">{frames}</div></div>'
        )
    return "".join(cards)


def _heat_tile(title: str, matrix: np.ndarray, order: np.ndarray, groups: np.ndarray, cell: int) -> Image.Image:
    sorted_matrix = np.clip(matrix[np.ix_(order, order)], 0.0, 1.0)[..., None]
    low, high = np.array([30, 40, 110]), np.array([250, 220, 60])
    pixels = (low + sorted_matrix * (high - low)).astype(np.uint8)
    body = Image.fromarray(pixels).resize((len(order) * cell,) * 2, Image.NEAREST)
    draw = ImageDraw.Draw(body)
    ordered = groups[order]
    for position in range(1, len(order)):
        if ordered[position] != ordered[position - 1]:
            draw.line([(position * cell, 0), (position * cell, body.height)], fill="white", width=2)
            draw.line([(0, position * cell), (body.width, position * cell)], fill="white", width=2)
    tile = Image.new("RGB", (body.width, body.height + 22), (30, 30, 30))
    tile.paste(body, (0, 22))
    ImageDraw.Draw(tile).text((4, 5), title, fill="white")
    return tile


def affinity_image(result: dict) -> np.ndarray:
    """The affinity maps with frames sorted by predicted scene. White lines mark scene borders."""
    groups = result["group_id"].numpy()
    order = np.lexsort((np.arange(len(groups)), groups))
    cell = max(10, min(40, 480 // max(len(groups), 1)))
    tiles = [_heat_tile(title, matrix, order, groups, cell) for title, matrix in result["maps"]]
    canvas = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), tiles[0].height), (30, 30, 30))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width + 8
    return np.asarray(canvas)


def run_bag(model, files, resolution, threshold, linkage, max_images, method, baseline_layer, baseline_threshold,
            match_patch_threshold=baseline.MATCH_PATCH_THRESHOLD, match_threshold=baseline.MATCH_GROUP_THRESHOLD,
            match_centered=True):
    paths = [str(f if isinstance(f, (str, Path)) else f.name) for f in (files or [])]
    if len(paths) < 2:
        raise gr.Error("Give at least 2 images.")
    if len(paths) > max_images:
        raise gr.Error(f"{len(paths)} images is more than --max-images {max_images}.")
    started = time.time()
    result = group_frames(model, paths, int(resolution), threshold, linkage, method, int(baseline_layer),
                          baseline_threshold, match_patch_threshold, match_threshold, match_centered)
    groups = result["group_id"]
    sizes = sorted((int((groups == g).sum()) for g in groups.unique()), reverse=True)
    summary = f"**{len(paths)} frames grouped into {len(sizes)} scenes** (sizes {sizes}) in {time.time() - started:.1f}s"
    return summary, scenes_html(paths, groups), affinity_image(result)


def build_ui(model: VGGTOmega, args) -> gr.Blocks:
    def group(files, resolution, threshold, linkage, method, baseline_layer, baseline_threshold,
              match_patch_threshold, match_threshold, match_centered):
        return run_bag(
            model, files, resolution, threshold, linkage, args.max_images, method, baseline_layer, baseline_threshold,
            match_patch_threshold, match_threshold, match_centered,
        )

    with gr.Blocks(title="Multi-scene VGGT-Omega") as demo:
        gr.Markdown(
            "# Multi-scene VGGT-Omega\nUpload frames from different scenes, in any order. "
            f"The model groups them into scenes. Device: `{next(model.parameters()).device}`."
        )
        files = gr.File(file_count="multiple", file_types=IMAGE_TYPES, label="Images (2 or more)")
        methods = [MATCHING, BASELINE] + ([LEARNED] if model.multiscene is not None else [])
        method = gr.Radio(methods, value=methods[0], label="Method")
        with gr.Row():
            resolution = gr.Slider(224, 640, value=args.image_resolution, step=16,
                                   label="Image resolution (not used by patch matching: 448 x 256)")
            threshold = gr.Slider(0.1, 0.9, value=args.threshold, step=0.05, label="Same-scene threshold (learned heads)")
            linkage = gr.Dropdown(["average", "single", "complete"], value=args.linkage, label="Linkage")
        with gr.Row():
            match_patch_threshold = gr.Slider(
                0.3, 0.95, value=baseline.MATCH_PATCH_THRESHOLD, step=0.05,
                label="Patch matching: patch threshold (a patch counts as matched at this cosine)",
            )
            match_threshold = gr.Slider(
                0.02, 0.6, value=baseline.MATCH_GROUP_THRESHOLD, step=0.01,
                label="Patch matching: same-scene share (frames join when this share of patches match)",
            )
            match_centered = gr.Checkbox(value=True, label="Patch matching: centering")
        with gr.Row():
            baseline_layer = gr.Slider(1, 24, value=args.baseline_layer, step=1,
                                       label="Backbone block (patch matching and patch cosine)")
            baseline_threshold = gr.Slider(
                0.5, 1.0, value=args.baseline_threshold, step=0.005, label="Baseline: cosine threshold"
            )
        button = gr.Button("Group", variant="primary")
        summary = gr.Markdown()
        scenes = gr.HTML()
        heatmap = gr.Image(label="Affinity, frames sorted by predicted scene")
        button.click(
            group,
            [files, resolution, threshold, linkage, method, baseline_layer, baseline_threshold,
             match_patch_threshold, match_threshold, match_centered],
            [summary, scenes, heatmap],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Local multi-scene grouping demo")
    parser.add_argument("--checkpoint", default="checkpoints/vggt_omega_1b_512.pt",
                        help="VGGT-Omega weights; a multi-scene training checkpoint also enables the learned heads")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--image-resolution", type=int, default=416, help="416 matches training")
    parser.add_argument("--route-layer", type=int, default=11, help="must match model.multiscene.route_layer in training")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--linkage", default="average", choices=["average", "single", "complete"])
    parser.add_argument("--baseline-layer", type=int, default=4, help="backbone block of the patch-cosine baseline")
    parser.add_argument("--baseline-threshold", type=float, default=0.9, help="fitted on retail, tune it for other data")
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--port", type=int, help="default: the first free port from 7860")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Loading {args.checkpoint} on {device} ...")
    model = load_model(args.checkpoint, device, args.route_layer, args.threshold, args.linkage)
    build_ui(model, args).queue(max_size=4).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
