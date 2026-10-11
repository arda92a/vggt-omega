"""Gradio demo: upload overlapping photos, get a 3D-aware panorama.

    uv run --extra demo python panorama/app.py --checkpoint checkpoints/vggt_omega_1b_512.pt

The model is loaded once. Its output for a set of photos is cached, so moving the render
sliders re-renders in seconds without running the network again.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import gradio as gr
import numpy as np
import torch

from stitch import BLENDS, CENTERS, PROJECTIONS, Settings, load_model, load_sky_session, reconstruct, render_panorama


class Engine:
    def __init__(self, checkpoint: str, device: str, skyseg: str) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Loading {checkpoint} on {self.device}")
        self.model = load_model(checkpoint, self.device)
        self.sky_session = load_sky_session(skyseg) if os.path.isfile(skyseg) else None
        if self.sky_session is None:
            print(f"Sky segmentation is off: {skyseg} or onnxruntime is missing.")
        self._key = None
        self._reconstruction = None

    def reconstruction(self, paths, resolution, color_side, use_sky, edge_rtol):
        key = (tuple((p, os.path.getmtime(p)) for p in paths), resolution, color_side, use_sky and self.sky_session is not None, edge_rtol)
        if key != self._key:
            self._reconstruction = None  # free the old one before building the next
            self._reconstruction = reconstruct(
                self.model,
                paths,
                self.device,
                resolution=resolution,
                color_long_side=color_side,
                sky_session=self.sky_session if use_sky else None,
                edge_rtol=edge_rtol,
            )
            self._key = key
        return self._reconstruction


def _paths(files) -> list[str]:
    return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]


def _heatmap(values: np.ndarray, mask: np.ndarray, colormap: int) -> np.ndarray:
    scaled = np.zeros(values.shape, np.uint8)
    if mask.any():
        low, high = np.percentile(values[mask], [2, 98])
        scaled = np.clip((values - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(scaled, colormap), cv2.COLOR_BGR2RGB)
    colored[~mask] = 0
    return colored


def build_ui(engine: Engine) -> gr.Blocks:
    def generate(files, projection, scale, center, blend, fill_gaps, conf_percentile, tolerance, feather, exposure, use_sky, resolution, color_side, progress=gr.Progress()):
        paths = _paths(files)
        if len(paths) < 2:
            raise gr.Error("Upload at least two overlapping photos.")
        settings = Settings(
            projection=projection,
            scale=scale,
            center=center,
            blend=blend,
            fill_gaps=fill_gaps,
            conf_percentile=conf_percentile,
            depth_tolerance=tolerance,
            feather=feather,
            exposure=exposure,
            use_sky=use_sky,
        )
        progress(0.05, desc="Reconstructing the 3D scene (cached for the same photos)")
        try:
            rec = engine.reconstruction(paths, int(resolution), int(color_side), use_sky, settings.edge_rtol)
            progress(0.6, desc="Rendering the panorama")
            pano = render_panorama(rec, settings)
        except ValueError as error:
            raise gr.Error(str(error)) from error

        covered = pano.alpha > 0
        views = _heatmap(pano.views.astype(np.float32), covered, cv2.COLORMAP_VIRIDIS)
        depth = np.log(np.where(np.isfinite(pano.depth), pano.depth, np.nan))
        depth_map = _heatmap(np.nan_to_num(depth), np.isfinite(depth), cv2.COLORMAP_TURBO)
        gallery = [(views, "photos blended per pixel"), (depth_map, "depth seen by the virtual camera (sky is black)")]

        rgba = np.dstack([pano.image, pano.alpha])
        path = os.path.join(tempfile.mkdtemp(prefix="panorama_"), "panorama.png")
        cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        return pano.image, path, gallery, pano.info

    with gr.Blocks(title="3D-aware panorama") as demo:
        gr.Markdown(
            "# 3D-aware panorama\n"
            "Photos go through VGGT-Omega for pose, depth and confidence. The scene is then rendered through a "
            "virtual camera with a z-buffer, so near and far objects are placed correctly even when the "
            "camera moved. Upload overlapping photos of the same scene, in one orientation."
        )
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Photos", file_count="multiple", file_types=["image"])
                preview = gr.Gallery(label="Uploaded", columns=4, height=180)
                projection = gr.Dropdown(
                    PROJECTIONS,
                    value=PROJECTIONS[0],
                    label="Projection",
                    info=(
                        "perspective: the camera turned on the spot (landscape, room). "
                        "plane: the camera walked along a shelf, wall or facade. The plane is looked at straight on, so "
                        "nothing stretches towards the edges and parallel lines stay parallel."
                    ),
                )
                scale = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Output scale (1.0 = detail of the photos)")
                blend = gr.Dropdown(
                    BLENDS,
                    value=BLENDS[0],
                    label="Blending",
                    info="multiband picks the best photo per pixel and joins them with a pyramid, so overlaps stay sharp. average mixes all photos.",
                )
                run = gr.Button("Generate panorama", variant="primary")
                with gr.Accordion("Rendering", open=False):
                    center = gr.Dropdown(CENTERS, value=CENTERS[0], label="Virtual camera position")
                    exposure = gr.Checkbox(True, label="Match exposure between photos")
                    fill_gaps = gr.Checkbox(True, label="Fill gaps from the best photo (no black holes between photos)")
                    use_sky = gr.Checkbox(engine.sky_session is not None, label="Treat sky as infinitely far", interactive=engine.sky_session is not None)
                    conf_percentile = gr.Slider(0, 60, value=15, step=1, label="Drop least confident pixels (%)")
                    tolerance = gr.Slider(0.02, 0.3, value=0.08, step=0.01, label="Depth agreement between photos (relative)")
                    feather = gr.Slider(0.02, 0.5, value=0.2, step=0.01, label="Seam softness (fraction of image size)")
                with gr.Accordion("Model input (re-runs the network)", open=False):
                    resolution = gr.Slider(256, 512, value=512, step=64, label="Model resolution")
                    color_side = gr.Slider(800, 3200, value=1600, step=100, label="Colour resolution (long side, px)")
            with gr.Column(scale=2):
                output = gr.Image(label="Panorama", type="numpy", format="png")
                download = gr.File(label="Download (PNG with transparent gaps)")
                diagnostics = gr.Gallery(label="Diagnostics", columns=2, height=260)
                info = gr.JSON(label="Details")

        files.change(lambda f: _paths(f), files, preview)
        run.click(
            generate,
            [files, projection, scale, center, blend, fill_gaps, conf_percentile, tolerance, feather, exposure, use_sky, resolution, color_side],
            [output, download, diagnostics, info],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="3D-aware panorama demo")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "vggt_omega_1b_512.pt"))
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--skyseg", default=str(ROOT / "skyseg.onnx"))
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    engine = Engine(args.checkpoint, args.device, args.skyseg)
    build_ui(engine).queue(max_size=4).launch(server_name=args.server_name, server_port=args.server_port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
