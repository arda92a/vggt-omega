"""Turn ReplicaRenderer output into the unified training layout.

ReplicaRenderer is the headless tool from facebookresearch/Replica-Dataset:

    ./build/bin/ReplicaRenderer mesh.ply textures glass.sur

It writes frame000000.jpg and depth000000.png in the working directory and does
not save poses. The motion is fixed in ReplicaSDK/src/render.cpp: start at
ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0), then each frame right-multiplies
the camera-from-world matrix by a -2.5 cm shift along camera x. Depth is
eye-space z times 65535 * 0.1, stored as uint16. Intrinsics match
ProjectionMatrixRDF_BottomLeft at 1280x960.

    uv run python training/dataset_preparation/convert_replica.py \
        ~/arda/vggt-omega/replica_data \
        ~/arda/vggt-omega/MERGED_DIR \
        --renderer ~/Replica-Dataset/ReplicaSDK/build/bin/ReplicaRenderer
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 960
DEPTH_SCALE = 65535.0 * 0.1
SHIFT_X = 0.025


def lookat_rdf(eye, target, up):
    """Camera-from-world 4x4. X right, Y down, Z forward, matching ModelViewLookAtRDF."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ np.asarray(eye, dtype=np.float64)
    return pose


def renderer_poses(count):
    """The exact trajectory ReplicaRenderer applies, one 3x4 pose per saved frame."""
    pose = lookat_rdf((0.0, 0.0, 4.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    step = np.eye(4)
    step[0, 3] = -SHIFT_X
    poses = []
    for _ in range(count):
        poses.append(pose[:3, :].astype(np.float32))
        pose = pose @ step
    return poses


def renderer_intrinsics():
    focal = WIDTH / 2.0
    return np.array(
        [
            [focal, 0.0, (WIDTH - 1.0) / 2.0],
            [0.0, focal, (HEIGHT - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def write_exr(path, depth):
    flags = []
    if hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_FLOAT"):
        flags += [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
    if hasattr(cv2, "IMWRITE_EXR_COMPRESSION") and hasattr(cv2, "IMWRITE_EXR_COMPRESSION_ZIP"):
        flags += [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]
    if not cv2.imwrite(str(path), np.ascontiguousarray(depth), flags):
        raise RuntimeError(f"failed to write {path}. This OpenCV build cannot write EXR.")
    loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if loaded is None or loaded.ndim != 2 or loaded.shape != depth.shape:
        raise RuntimeError(f"{path} read back as {None if loaded is None else loaded.shape}")


def discover_scenes(root):
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "mesh.ply").is_file())


def output_name(folder):
    return folder if folder.startswith("replica_") else f"replica_{folder}"


def already_done(dest, overwrite):
    if overwrite or not dest.is_dir():
        return False
    needed = ["image_names.json", "cam_from_worlds.npy", "intrinsics.npy", "images", "depths"]
    return all((dest / name).exists() for name in needed)


def depth_png_to_metres(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"could not read {path}")
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    depth = raw.astype(np.float32) / DEPTH_SCALE
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return depth


def render_scene(renderer, scene):
    work = scene / ".replica_render"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    command = [str(renderer), str((scene / "mesh.ply").resolve()), str((scene / "textures").resolve())]
    glass = scene / "glass.sur"
    if glass.is_file():
        command.append(str(glass.resolve()))
    completed = subprocess.run(command, cwd=work, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {completed.returncode}"
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(detail)
    frames = sorted(work.glob("frame*.jpg"))
    if not frames:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"ReplicaRenderer wrote no frames in {work}")
    return work, frames


def publish(dest, work, frames):
    poses = renderer_poses(len(frames))
    intrinsics = renderer_intrinsics()
    staging = dest.parent / f".{dest.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    images = staging / "images"
    depths = staging / "depths"
    images.mkdir(parents=True)
    depths.mkdir()
    names = []
    coverages = []
    try:
        for index, frame in enumerate(frames):
            name = f"{index:05d}.jpg"
            image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
            if image is None or image.shape[1] != WIDTH or image.shape[0] != HEIGHT:
                raise RuntimeError(f"{frame.name} is {None if image is None else image.shape}, expected {HEIGHT}x{WIDTH}")
            shutil.copy2(frame, images / name)
            depth_path = work / f"depth{index:06d}.png"
            depth = depth_png_to_metres(depth_path)
            if depth.shape != (HEIGHT, WIDTH):
                raise RuntimeError(f"{depth_path.name} is {depth.shape}")
            write_exr(depths / f"{index:05d}.exr", depth)
            names.append(name)
            coverages.append(float((depth > 0).mean()))
        (staging / "image_names.json").write_text(json.dumps(names, indent=2) + "\n")
        np.save(staging / "cam_from_worlds.npy", np.stack(poses))
        np.save(staging / "intrinsics.npy", np.stack([intrinsics for _ in poses]))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(work, ignore_errors=True)
    if dest.exists():
        backup = dest.parent / f".{dest.name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        dest.rename(backup)
        staging.rename(dest)
        shutil.rmtree(backup)
    else:
        staging.rename(dest)
    return float(np.mean(coverages)) if coverages else 0.0


def self_test():
    poses = renderer_poses(3)
    origin = np.zeros(3)
    first = poses[0][:, :3] @ origin + poses[0][:, 3]
    assert np.allclose(first, [0, 0, 4], atol=1e-6), first
    second = poses[1][:, :3] @ origin + poses[1][:, 3]
    assert abs(float(second[0]) + SHIFT_X) < 1e-6, second
    assert abs(np.linalg.det(poses[0][:, :3]) - 1.0) < 1e-6
    matrix = renderer_intrinsics()
    assert matrix[0, 0] == WIDTH / 2.0 and matrix[0, 2] == (WIDTH - 1) / 2.0
    print("self-test passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", nargs="?", type=Path, help="Replica root, one folder per scene")
    parser.add_argument("dst", nargs="?", type=Path, help="unified root; replica_* folders are added beside existing scenes")
    parser.add_argument("--renderer", type=Path, default=os.environ.get("REPLICA_RENDERER"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.src is None or args.dst is None:
        parser.error("src and dst are required")
    if args.renderer is None or not Path(args.renderer).is_file():
        raise SystemExit(
            "ReplicaRenderer binary not found. Build it, then pass --renderer:\n"
            "  git clone https://github.com/facebookresearch/Replica-Dataset.git\n"
            "  cd Replica-Dataset/ReplicaSDK && mkdir build && cd build && cmake .. && make -j\n"
            "  --renderer Replica-Dataset/ReplicaSDK/build/bin/ReplicaRenderer"
        )

    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.is_dir():
        raise SystemExit(f"src is not a directory: {src}")
    if dst == src or src in dst.parents:
        raise SystemExit("dst must be outside src")
    dst.mkdir(parents=True, exist_ok=True)
    scenes = discover_scenes(src)
    if args.scenes:
        chosen = set(args.scenes)
        scenes = [scene for scene in scenes if scene.name in chosen]
    if not scenes:
        raise SystemExit(f"no scene folders with mesh.ply under {src}")
    if args.dry_run:
        print(f"{len(scenes)} scenes: " + ", ".join(output_name(scene.name) for scene in scenes))
        return 0

    report = []
    failed = 0
    for scene in scenes:
        dest = dst / output_name(scene.name)
        if already_done(dest, args.overwrite):
            print(f"skip {dest.name}")
            report.append({"scene": dest.name, "source": scene.name, "status": "skipped"})
            continue
        print(f"rendering {scene.name} -> {dest.name}")
        try:
            work, frames = render_scene(args.renderer, scene)
            coverage = publish(dest, work, frames)
        except Exception as exc:
            failed += 1
            print(f"FAIL {scene.name}: {exc}")
            report.append({"scene": dest.name, "source": scene.name, "status": "failed", "error": str(exc)})
            continue
        report.append(
            {
                "scene": dest.name,
                "source": scene.name,
                "frames": len(frames),
                "depth_coverage": coverage,
                "status": "ok",
            }
        )
        print(f"ok {dest.name}: {len(frames)} frames, depth covers {coverage * 100:.1f}%")

    (dst / "replica_convert_report.json").write_text(json.dumps(report, indent=2) + "\n")
    ok = sum(item["status"] == "ok" for item in report)
    print(f"{ok} converted, {failed} failed, {len(report) - ok - failed} skipped")
    if failed:
        return 1
    print("Next: python training/dataset_preparation/split_sequences.py", dst, "--val-per-source 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
