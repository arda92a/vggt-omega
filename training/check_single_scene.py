"""Check that a single scene still matches the original VGGT-Omega forward.

The multi-scene block is zero at initialization, and one scene is a single
group, so pose and depth should match the checkpoint before any fine-tune.
Run this on the GPU machine once the checkpoint and a few images are there.

    python check_single_scene.py \
      --checkpoint /path/to/vggt_omega_1b_512.pt \
      --images /path/a.png /path/b.png /path/c.png
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggt_omega.models import VGGTOmega


COMPARE_KEYS = ("pose_enc", "depth")


def prediction_gap(reference: dict, candidate: dict) -> dict[str, dict[str, float]]:
    """Max and mean absolute difference for pose and depth."""
    report = {}
    for key in COMPARE_KEYS:
        if key not in reference or key not in candidate:
            raise KeyError(f"Both forwards must contain '{key}'")
        difference = (reference[key].float() - candidate[key].float()).abs()
        report[key] = {"max": float(difference.max()), "mean": float(difference.mean())}
    return report


def unwrap_state_dict(checkpoint) -> dict:
    """Released checkpoints are bare state dicts. Trainer files store them under 'model'."""
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a state dict")
    return checkpoint


def load_pair(checkpoint_path: str, device: str):
    """Original model, and the same weights with an untrained multi-scene block."""
    state = unwrap_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    reference = VGGTOmega().to(device).eval()
    reference.load_state_dict(state, strict=True)

    candidate = VGGTOmega(
        multiscene={
            "enabled": True,
            "isolate_camera": True,
            "camera_groups": "predicted",
            "num_heads": 16,
        }
    ).to(device).eval()
    missing, unexpected = candidate.load_state_dict(state, strict=False)
    foreign = [key for key in missing if not key.startswith("multiscene.")]
    if foreign or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the multi-scene model. "
            f"Unexpected missing keys: {foreign[:10]}. Unexpected extra keys: {unexpected[:10]}."
        )
    return reference, candidate


def compare_checkpoint(checkpoint_path: str, images: torch.Tensor, device: str, atol: float) -> dict:
    reference_model, candidate_model = load_pair(checkpoint_path, device)
    images = images.to(device)
    with torch.inference_mode():
        reference = reference_model(images)
        mixed_off = _forward_isolated(candidate_model, images, isolate=False)
        one_group = _forward_isolated(candidate_model, images, isolate=True)
    report = {
        "block_disabled": prediction_gap(reference, mixed_off),
        "one_group": prediction_gap(reference, one_group),
    }
    report["ok"] = _within(report, atol)
    return report


def _forward_isolated(model, images, isolate: bool):
    previous = model.isolate_camera
    model.isolate_camera = isolate
    try:
        return model(images)
    finally:
        model.isolate_camera = previous


def _within(report: dict, atol: float) -> bool:
    for mode in ("block_disabled", "one_group"):
        for key in COMPARE_KEYS:
            if report[mode][key]["max"] > atol:
                return False
    return True


def _parse_args():
    parser = argparse.ArgumentParser(description="Single-scene agreement with the original checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.images:
        raise SystemExit("Pass at least two images. A single-scene check needs a real forward.")
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(args.images, image_resolution=args.image_resolution)
    report = compare_checkpoint(args.checkpoint, images, args.device, args.atol)
    for mode in ("block_disabled", "one_group"):
        for key, values in report[mode].items():
            print(f"{mode} {key}: max {values['max']:.3e}  mean {values['mean']:.3e}")
    if report["ok"]:
        print(f"single-scene check passed (atol {args.atol:g})")
        return 0
    print(f"single-scene check failed (atol {args.atol:g})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
