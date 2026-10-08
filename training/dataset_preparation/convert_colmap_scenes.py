"""Convert ETH3D-style scene folders into the unified layout the loader reads.

Each source scene is a directory with images and a COLMAP text model:

    <src>/<scene>/
      images/
      dslr_calibration_undistorted/
        cameras.txt
        images.txt
        points3D.txt

The undistorted package has no dense depth. This script rasterizes points3D.txt
into a z-depth EXR per frame (invalid pixels stay 0). COLMAP already stores
camera-from-world poses in the OpenCV frame (x right, y down, z forward), in
metres for ETH3D, so rotation and translation are copied through.

    python training/dataset_preparation/convert_colmap_scenes.py \
        /path/to/UNIFIED_DIR /path/to/MERGED_DIR --prefix eth3d
    python training/dataset_preparation/split_sequences.py /path/to/MERGED_DIR

Write MERGED_DIR outside the raw root. Scene folders are named eth3d_<scene> so
split_sequences.py can hold out whole families (relief and relief_2 stay together).
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# OpenCV only writes and reads EXR when this is set before it is imported.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from PIL import Image

def quat_to_rotmat(qw, qx, qy, qz):
    """COLMAP Hamilton quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix."""
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"degenerate quaternion {(qw, qx, qy, qz)}")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def intrinsics_from_camera(model, params):
    """Pinhole K from a COLMAP camera. Nonzero distortion is rejected: the target has none."""
    model = model.upper()
    params = [float(value) for value in params]
    distortion = []
    if model == "SIMPLE_PINHOLE":
        focal, cx, cy = params[:3]
        fx = fy = focal
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_RADIAL":
        focal, cx, cy = params[:3]
        fx = fy = focal
        distortion = params[3:4]
    elif model == "RADIAL":
        focal, cx, cy = params[:3]
        fx = fy = focal
        distortion = params[3:5]
    elif model in {"OPENCV", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
        distortion = params[4:]
    else:
        raise ValueError(f"unsupported COLMAP camera model {model}")
    if any(abs(value) > 1e-8 for value in distortion):
        raise ValueError(
            f"camera model {model} still has distortion {distortion}. "
            "Point this script at dslr_calibration_undistorted, whose images are already rectified."
        )
    if fx <= 0 or fy <= 0:
        raise ValueError(f"non-positive focal length fx={fx} fy={fy}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def iter_model_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if raw.startswith("#"):
                continue
            yield raw.rstrip("\n")


def read_cameras(path):
    cameras = {}
    for line in iter_model_lines(path):
        parts = line.split()
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "K": intrinsics_from_camera(model, parts[4:]),
        }
    if not cameras:
        raise ValueError(f"no cameras in {path}")
    return cameras


def read_images(path):
    """Return frames in file order. The following POINTS2D line is not used."""
    lines = list(iter_model_lines(path))
    if len(lines) % 2 != 0:
        raise ValueError(f"{path} should alternate a pose line and a POINTS2D line")
    frames = []
    for index in range(0, len(lines), 2):
        parts = lines[index].split()
        if len(parts) < 10:
            raise ValueError(f"short image line in {path}: {lines[index]}")
        rotation = quat_to_rotmat(*map(float, parts[1:5]))
        translation = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        pose = np.concatenate([rotation, translation[:, None]], axis=1)
        frames.append(
            {
                "image_id": int(parts[0]),
                "camera_id": int(parts[8]),
                "name": " ".join(parts[9:]),
                "cam_from_world": pose,
            }
        )
    if not frames:
        raise ValueError(f"no images in {path}")
    return frames


def read_points_by_image(path):
    """3D points grouped by the images that observe them. Missing file -> no points."""
    if not path.is_file():
        return {}
    buckets = defaultdict(list)
    for line in iter_model_lines(path):
        parts = line.split()
        if len(parts) < 8:
            continue
        xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
        # POINT3D_ID X Y Z R G B ERROR, then (IMAGE_ID, POINT2D_IDX) pairs.
        track = parts[8:]
        for cursor in range(0, len(track) - 1, 2):
            buckets[int(track[cursor])].append(xyz)
    return {image_id: np.asarray(points, dtype=np.float64) for image_id, points in buckets.items()}


def find_image(scene, name, images_dirname):
    basename = Path(name).name
    for candidate in (
        scene / name,
        scene / images_dirname / name,
        scene / images_dirname / basename,
    ):
        if candidate.is_file():
            return candidate
    return None


def image_size(path):
    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
        if orientation not in (0, 1):
            raise ValueError(
                f"{path} has EXIF orientation {orientation}. "
                "Bake that rotation into the pixels before converting."
            )
        return image.size  # (width, height)


def check_rotation(pose, name):
    rotation = pose[:, :3]
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} pose is not finite")
    if abs(np.linalg.det(rotation) - 1.0) > 1e-3:
        raise ValueError(f"{name} rotation determinant is {np.linalg.det(rotation):.4f}, expected 1")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation is not orthonormal")


def scale_intrinsics(intrinsics, scale_x, scale_y):
    scaled = intrinsics.copy()
    scaled[0, :] *= scale_x
    scaled[1, :] *= scale_y
    return scaled


def rasterize_depth(height, width, intrinsics, pose, points, splat_radius):
    """Z-depth of observed points. Pixel (col, row) covers [col, col+1) x [row, row+1)."""
    depth = np.zeros((height, width), dtype=np.float32)
    if points is None or len(points) == 0:
        return depth
    camera = points @ pose[:, :3].T + pose[:, 3]
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z > 1e-6)
    if not np.any(valid):
        return depth
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u = fx * camera[:, 0] / z + cx
    v = fy * camera[:, 1] / z + cy
    cols = np.floor(u).astype(np.int32)
    rows = np.floor(v).astype(np.int32)
    valid &= (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not np.any(valid):
        return depth
    cols, rows, z = cols[valid], rows[valid], z[valid].astype(np.float32)
    # Farther points first, so a closer point keeps the pixel.
    order = np.argsort(-z)
    cols, rows, z = cols[order], rows[order], z[order]
    radius = int(splat_radius)
    if radius <= 0:
        depth[rows, cols] = z
        return depth
    for row, col, value in zip(rows, cols, z):
        r0 = max(0, int(row) - radius)
        r1 = min(height, int(row) + radius + 1)
        c0 = max(0, int(col) - radius)
        c1 = min(width, int(col) + radius + 1)
        depth[r0:r1, c0:c1] = value
    return depth


def write_exr(path, depth):
    flags = []
    if hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_FLOAT"):
        flags += [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
    if hasattr(cv2, "IMWRITE_EXR_COMPRESSION") and hasattr(cv2, "IMWRITE_EXR_COMPRESSION_ZIP"):
        flags += [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]
    if not cv2.imwrite(str(path), np.ascontiguousarray(depth), flags):
        raise RuntimeError(
            f"failed to write {path}. This OpenCV build cannot write EXR "
            "(OPENCV_IO_ENABLE_OPENEXR=1 must be set before importing cv2)."
        )
    loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if loaded is None or loaded.ndim != 2 or loaded.shape != depth.shape:
        raise RuntimeError(
            f"{path} read back as {None if loaded is None else loaded.shape}, expected {depth.shape}. "
            "The loader requires a single-channel 2D EXR."
        )


def place_image(source, dest, mode, size):
    """Copy, link, or resize one image. size is (width, height) after an optional downscale."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if mode == "symlink":
            dest.symlink_to(source.resolve())
            return
        if mode == "hardlink":
            try:
                os.link(source, dest)
                return
            except OSError:
                pass
        shutil.copy2(source, dest)
        return

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read {source}")
    width, height = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    suffix = dest.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(str(dest), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif suffix == ".png":
        ok = cv2.imwrite(str(dest), resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        dest = dest.with_suffix(".jpg")
        ok = cv2.imwrite(str(dest), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"could not write {dest}")


def output_name(source_name, used):
    basename = Path(source_name).name
    if basename not in used and Path(basename).stem not in {Path(name).stem for name in used}:
        used.add(basename)
        return basename
    stem = Path(basename).stem
    suffix = Path(basename).suffix
    index = 1
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def convert_scene(scene, dest, args):
    model_dir = scene / args.calibration_dirname
    frames = read_images(model_dir / "images.txt")
    cameras = read_cameras(model_dir / "cameras.txt")
    points = read_points_by_image(model_dir / "points3D.txt")

    usable = []
    missing = []
    for frame in frames:
        path = find_image(scene, frame["name"], args.images_dirname)
        if path is None:
            missing.append(frame["name"])
            continue
        if frame["camera_id"] not in cameras:
            raise ValueError(f"{frame['name']} references missing camera {frame['camera_id']}")
        check_rotation(frame["cam_from_world"], frame["name"])
        frame["path"] = path
        usable.append(frame)
    if len(usable) < 3:
        raise ValueError(f"only {len(usable)} images on disk, need at least 3. Missing e.g. {missing[:3]}")

    usable.sort(key=lambda frame: (Path(frame["name"]).name, frame["name"]))
    names_used = set()
    records = []
    for frame in usable:
        camera = cameras[frame["camera_id"]]
        width, height = image_size(frame["path"])
        intrinsics = camera["K"].copy()
        if (width, height) != (camera["width"], camera["height"]):
            intrinsics = scale_intrinsics(
                intrinsics,
                width / camera["width"],
                height / camera["height"],
            )
        stored_size = None
        if args.max_side and max(width, height) > args.max_side:
            scale = args.max_side / max(width, height)
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            intrinsics = scale_intrinsics(intrinsics, new_width / width, new_height / height)
            width, height = new_width, new_height
            stored_size = (width, height)
        if intrinsics.min() < 0 or not np.isfinite(intrinsics).all():
            raise ValueError(f"{frame['name']} intrinsics are negative or not finite")
        if not (0 <= intrinsics[0, 2] < width and 0 <= intrinsics[1, 2] < height):
            raise ValueError(f"{frame['name']} principal point is outside the stored image")
        name = output_name(frame["name"], names_used)
        if stored_size is not None and Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            name = f"{Path(name).stem}.jpg"
        records.append(
            {
                "frame": frame,
                "name": name,
                "intrinsics": intrinsics,
                "width": width,
                "height": height,
                "stored_size": stored_size,
            }
        )

    staging = dest.parent / f".{dest.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    images_dir = staging / "images"
    depths_dir = staging / "depths"
    images_dir.mkdir(parents=True)
    depths_dir.mkdir()

    coverages = []
    try:
        for record in records:
            place_image(record["frame"]["path"], images_dir / record["name"], args.link, record["stored_size"])
            depth = rasterize_depth(
                record["height"],
                record["width"],
                record["intrinsics"],
                record["frame"]["cam_from_world"],
                points.get(record["frame"]["image_id"]),
                args.splat_radius,
            )
            write_exr(depths_dir / f"{Path(record['name']).stem}.exr", depth)
            coverages.append(float((depth > 0).mean()))
        image_names = [record["name"] for record in records]
        (staging / "image_names.json").write_text(json.dumps(image_names, indent=2) + "\n")
        np.save(
            staging / "cam_from_worlds.npy",
            np.stack([record["frame"]["cam_from_world"] for record in records]).astype(np.float32),
        )
        np.save(
            staging / "intrinsics.npy",
            np.stack([record["intrinsics"] for record in records]).astype(np.float32),
        )
        (staging / "source_frames.json").write_text(
            json.dumps(
                [
                    {
                        "image": record["name"],
                        "source": record["frame"]["name"],
                        "image_id": record["frame"]["image_id"],
                    }
                    for record in records
                ],
                indent=2,
            )
            + "\n"
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if dest.exists():
        backup = dest.parent / f".{dest.name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        dest.rename(backup)
        staging.rename(dest)
        shutil.rmtree(backup)
    else:
        staging.rename(dest)

    return {
        "scene": dest.name,
        "source": scene.name,
        "frames": len(records),
        "missing_images": missing,
        "depth_coverage": float(np.mean(coverages)) if coverages else 0.0,
    }


def scene_output_name(folder, prefix):
    if prefix and not folder.startswith(prefix + "_"):
        return f"{prefix}_{folder}"
    return folder


def already_done(dest, overwrite):
    if overwrite or not dest.is_dir():
        return False
    needed = ["image_names.json", "cam_from_worlds.npy", "intrinsics.npy", "images", "depths"]
    return all((dest / name).exists() for name in needed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", nargs="?", type=Path, help="raw root, one folder per scene")
    parser.add_argument("dst", nargs="?", type=Path, help="unified root to create; must not sit inside src")
    parser.add_argument("--prefix", default="eth3d", help="scene folder prefix, eth3d_courtyard")
    parser.add_argument("--calibration-dirname", default="dslr_calibration_undistorted")
    parser.add_argument("--images-dirname", default="images")
    parser.add_argument("--link", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--max-side", type=int, default=0, help="downscale the long side before storing; 0 keeps native")
    parser.add_argument(
        "--splat-radius",
        type=int,
        default=0,
        help="paint each sparse point as a square of this radius so nearest-neighbor resize keeps it",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="only these source folder names")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="run a synthetic pose and depth check, then exit")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if args.src is None or args.dst is None:
        parser.error("src and dst are required")

    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.is_dir():
        raise SystemExit(f"src is not a directory: {src}")
    if dst == src or src in dst.parents:
        raise SystemExit("dst must be outside src so converted folders are not read as raw scenes")
    dst.mkdir(parents=True, exist_ok=True)

    scenes = []
    for path in sorted(src.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if args.scenes and path.name not in set(args.scenes):
            continue
        if (path / args.calibration_dirname / "cameras.txt").is_file():
            scenes.append(path)
    if not scenes:
        raise SystemExit(f"no scenes with {args.calibration_dirname}/cameras.txt under {src}")

    report = []
    failed = 0
    for scene in scenes:
        name = scene_output_name(scene.name, args.prefix)
        dest = dst / name
        if already_done(dest, args.overwrite):
            print(f"skip {name} (already converted)")
            report.append({"scene": name, "source": scene.name, "status": "skipped"})
            continue
        try:
            summary = convert_scene(scene, dest, args)
        except Exception as exc:
            failed += 1
            print(f"FAIL {scene.name}: {exc}")
            report.append({"scene": name, "source": scene.name, "status": "failed", "error": str(exc)})
            continue
        summary["status"] = "ok"
        report.append(summary)
        missing = f", {len(summary['missing_images'])} images missing" if summary["missing_images"] else ""
        print(
            f"ok {name}: {summary['frames']} frames, "
            f"sparse depth covers {summary['depth_coverage'] * 100:.3f}% of pixels{missing}"
        )

    (dst / "convert_report.json").write_text(json.dumps(report, indent=2) + "\n")
    ok = sum(1 for item in report if item["status"] == "ok")
    print(f"{ok} converted, {failed} failed, {len(report) - ok - failed} skipped")
    sparse = [
        item["scene"]
        for item in report
        if item["status"] == "ok" and item.get("depth_coverage", 1) < 0.01
    ]
    if sparse:
        noun = "scene has" if len(sparse) == 1 else "scenes have"
        print(
            f"{len(sparse)} {noun} sparse depth on under 1% of pixels. "
            "Stage 1 does not use depth. For stage 2, rerun with --max-side 1024 --splat-radius 2, "
            "or replace depths/ with dense laser-scan depth."
        )
    if failed:
        return 1
    print("Next: python training/dataset_preparation/split_sequences.py", dst)
    return 0


def self_test():
    """Identity pose projects (0, 0, 10) onto the pixel whose center is the principal point."""
    rotation = quat_to_rotmat(1, 0, 0, 0)
    assert np.allclose(rotation, np.eye(3))
    quarter = np.sqrt(0.5)
    yaw90 = quat_to_rotmat(quarter, 0, 0, quarter)
    assert np.allclose(yaw90, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), atol=1e-6)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scene = root / "raw" / "toy"
        images = scene / "images"
        model = scene / "dslr_calibration_undistorted"
        images.mkdir(parents=True)
        model.mkdir()
        width = height = 11
        fx = fy = 10.0
        cx = cy = 5.5
        (model / "cameras.txt").write_text(
            f"# Camera list\n1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n"
        )
        # Second camera is translated to (1, 0, 0): t = -R C.
        (model / "images.txt").write_text(
            "# Image list\n"
            "1 1 0 0 0 0 0 0 1 view_a.png\n"
            "\n"
            "2 1 0 0 0 -1 0 0 1 view_b.png\n"
            "\n"
            "3 1 0 0 0 0 0 0 1 view_c.png\n"
            "\n"
        )
        # Point on the optical axis, observed by every view. One extra point behind the camera.
        (model / "points3D.txt").write_text(
            "# 3D point list\n"
            "1 0 0 10 0 0 0 0 1 0 2 0 3 0\n"
            "2 0 0 -5 0 0 0 0 1 1\n"
        )
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for name in ("view_a.png", "view_b.png", "view_c.png"):
            cv2.imwrite(str(images / name), blank)

        status = main([str(root / "raw"), str(root / "out"), "--prefix", "eth3d", "--link", "copy"])
        if status != 0:
            raise SystemExit("self-test conversion failed")
        out = root / "out" / "eth3d_toy"
        names = json.loads((out / "image_names.json").read_text())
        poses = np.load(out / "cam_from_worlds.npy")
        intrinsics = np.load(out / "intrinsics.npy")
        assert names == ["view_a.png", "view_b.png", "view_c.png"]
        assert poses.shape == (3, 3, 4) and intrinsics.shape == (3, 3, 3)
        depth_a = cv2.imread(str(out / "depths" / "view_a.exr"), cv2.IMREAD_UNCHANGED)
        depth_b = cv2.imread(str(out / "depths" / "view_b.exr"), cv2.IMREAD_UNCHANGED)
        # u = 5.5 lands in pixel 5, whose center is 5.5.
        assert depth_a.shape == (11, 11)
        assert abs(float(depth_a[5, 5]) - 10.0) < 1e-4, depth_a[5, 5]
        assert float(depth_a[0, 0]) == 0.0
        # view_b sees the point at u = 4.5, v = 5.5, still z = 10.
        assert abs(float(depth_b[5, 4]) - 10.0) < 1e-4, depth_b[5, 4]
        assert np.allclose(poses[1, :, 3], [-1, 0, 0])
    print("self-test passed")


if __name__ == "__main__":
    sys.exit(main())
