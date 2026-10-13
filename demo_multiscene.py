"""Local Gradio demo: upload frames from any scenes and see how the multi-scene VGGT-Omega groups them.

    uv run --extra ui python demo_multiscene.py --checkpoint checkpoints/multiscene.pt
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
from vggt_omega.utils.load_fn import load_and_preprocess_images

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
        raise SystemExit(f"{checkpoint} has no multiscene.* weights. Use a checkpoint from train.py.")
    model = VGGTOmega(
        embed_dim=state["aggregator.camera_token"].shape[-1],
        autocast=device.type == "cuda",
        multiscene={
            "enabled": True,
            "num_heads": 16,
            "threshold": threshold,
            "linkage": linkage,
            "route_layer": route_layer,
            "isolate_backbone": True,
            "isolate_camera": True,
        },
    )
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def group_frames(model: VGGTOmega, paths: list[str], resolution: int, threshold: float, linkage: str) -> dict:
    device = next(model.parameters()).device
    images = load_and_preprocess_images(paths, mode="balanced", image_resolution=resolution).to(device)
    model.multiscene.threshold = float(threshold)
    model.multiscene.linkage = linkage
    prediction = model(images)
    return {key: prediction[key][0].float().cpu() for key in ("affinity", "route_affinity", "group_id", "route_group_id")}


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
    """Final and route affinity with frames sorted by predicted scene. White lines mark scene borders."""
    groups = result["group_id"].numpy()
    order = np.lexsort((np.arange(len(groups)), groups))
    cell = max(10, min(40, 480 // max(len(groups), 1)))
    tiles = [
        _heat_tile("final affinity", result["affinity"].numpy(), order, groups, cell),
        _heat_tile("route affinity", result["route_affinity"].numpy(), order, groups, cell),
    ]
    canvas = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), tiles[0].height), (30, 30, 30))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width + 8
    return np.asarray(canvas)


def run_bag(model, files, resolution, threshold, linkage, max_images):
    paths = [str(f if isinstance(f, (str, Path)) else f.name) for f in (files or [])]
    if len(paths) < 2:
        raise gr.Error("Give at least 2 images.")
    if len(paths) > max_images:
        raise gr.Error(f"{len(paths)} images is more than --max-images {max_images}.")
    started = time.time()
    result = group_frames(model, paths, int(resolution), threshold, linkage)
    groups = result["group_id"]
    sizes = sorted((int((groups == g).sum()) for g in groups.unique()), reverse=True)
    summary = f"**{len(paths)} frames grouped into {len(sizes)} scenes** (sizes {sizes}) in {time.time() - started:.1f}s"
    return summary, scenes_html(paths, groups), affinity_image(result)


def build_ui(model: VGGTOmega, args) -> gr.Blocks:
    def group(files, resolution, threshold, linkage):
        return run_bag(model, files, resolution, threshold, linkage, args.max_images)

    with gr.Blocks(title="Multi-scene VGGT-Omega") as demo:
        gr.Markdown(
            "# Multi-scene VGGT-Omega\nUpload frames from different scenes, in any order. "
            f"The model groups them into scenes. Device: `{next(model.parameters()).device}`."
        )
        files = gr.File(file_count="multiple", file_types=IMAGE_TYPES, label="Images (2 or more)")
        with gr.Row():
            resolution = gr.Slider(224, 640, value=args.image_resolution, step=16, label="Image resolution")
            threshold = gr.Slider(0.1, 0.9, value=args.threshold, step=0.05, label="Same-scene threshold")
            linkage = gr.Dropdown(["average", "single", "complete"], value=args.linkage, label="Linkage")
        button = gr.Button("Group", variant="primary")
        summary = gr.Markdown()
        scenes = gr.HTML()
        heatmap = gr.Image(label="Affinity, frames sorted by predicted scene")
        button.click(group, [files, resolution, threshold, linkage], [summary, scenes, heatmap])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Local multi-scene grouping demo")
    parser.add_argument("--checkpoint", required=True, help="checkpoint.pt copied from the training server")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--image-resolution", type=int, default=416, help="416 matches training")
    parser.add_argument("--route-layer", type=int, default=11, help="must match model.multiscene.route_layer in training")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--linkage", default="average", choices=["average", "single", "complete"])
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Loading {args.checkpoint} on {device} ...")
    model = load_model(args.checkpoint, device, args.route_layer, args.threshold, args.linkage)
    build_ui(model, args).queue(max_size=4).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
