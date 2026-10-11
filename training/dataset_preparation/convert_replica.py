"""Render Replica meshes into the unified layout and drop them next to ETH3D.

The Replica download is one mesh per scene (mesh.ply + PTex textures), not images.
This script flies a camera through each mesh and writes:

    <dst>/replica_<scene>/
      images/00000.jpg
      depths/00000.exr          # OpenCV z-depth, metres, invalid = 0
      image_names.json
      cam_from_worlds.npy       # (N, 3, 4), X_cam = R @ X_world + t
      intrinsics.npy            # (N, 3, 3)

Replica is Z-up. Habitat's camera is OpenGL (x right, y up, z backward); the saved
pose flips y and z into the OpenCV frame the loader expects.

    uv run python training/dataset_preparation/convert_replica.py \
        ~/arda/vggt-omega/replica_data \
        ~/arda/vggt-omega/MERGED_DIR \
        --frames 100
    uv run python training/dataset_preparation/split_sequences.py \
        ~/arda/vggt-omega/MERGED_DIR --val-per-source 4

Needs habitat-sim with PTex still enabled (the 0.2.x line). Newer habitat builds
dropped that shader and render flat colour; this script rejects those frames
instead of writing them into the training set.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def quat_to_rotmat(qw, qx, qy, qz):
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def opencv_from_opengl_c2w(rotation_c2w, translation):
    """OpenGL camera-to-world pose to an OpenCV camera-from-world 3x4 matrix."""
    flip = np.diag([1.0, -1.0, -1.0])
    rotation = flip @ np.asarray(rotation_c2w, dtype=np.float64).T
    translation = -rotation @ np.asarray(translation, dtype=np.float64)
    return np.concatenate([rotation, translation.reshape(3, 1)], axis=1)


def look_rotation(yaw, pitch):
    """OpenGL camera-to-world rotation. Replica up is +Z, and yaw 0 looks along +Y."""
    forward = np.array(
        [
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    x_axis = np.cross(UP, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("look direction is parallel to the up axis")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def pinhole(width, height, hfov_deg):
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


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


def _quat_parts(rotation):
    if hasattr(rotation, "w"):
        return float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)
    vector = rotation.vector
    return float(rotation.scalar), float(vector.x), float(vector.y), float(vector.z)


def make_simulator(scene, width, height, hfov):
    import habitat_sim
    import quaternion

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str((scene / "mesh.ply").resolve())
    sim_cfg.gpu_device_id = 0
    sim_cfg.enable_physics = False

    sensors = []
    for uuid, kind in (("color", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = kind
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [height, width]
        spec.hfov = float(hfov)
        spec.position = [0.0, 0.0, 0.0]
        sensors.append(spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    # The eye position is set explicitly, so the agent must not add another height offset.
    agent_cfg.height = 0.0
    agent_cfg.radius = 0.1

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    navmesh = scene / "habitat" / "mesh_semantic.navmesh"
    if navmesh.is_file() and not sim.pathfinder.is_loaded:
        sim.pathfinder.load_nav_mesh(str(navmesh))
    if not sim.pathfinder.is_loaded:
        sim.close()
        raise RuntimeError(f"{scene.name} has no loaded navmesh at {navmesh}")
    return sim, quaternion


def _prepare_depth(depth, height, width):
    depth = np.squeeze(np.asarray(depth)).astype(np.float32)
    if depth.shape != (height, width):
        raise RuntimeError(f"depth shape {depth.shape}, expected {(height, width)}")
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return depth


def _prepare_color(color):
    rgb = np.asarray(color)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 1 if rgb.max() <= 1.5 else 255)
        if rgb.max() <= 1.5:
            rgb = rgb * 255.0
        rgb = rgb.astype(np.uint8)
    return rgb


def sample_views(sim, quaternion_module, args, rng):
    agent = sim.get_agent(0)
    views = []
    attempts = 0
    limit = args.frames * 30
    while len(views) < args.frames and attempts < limit:
        attempts += 1
        floor = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
        if not np.isfinite(floor).all():
            continue
        eye = floor + UP * args.height
        yaw = float(rng.uniform(-np.pi, np.pi))
        pitch = float(rng.uniform(-args.pitch, args.pitch))
        rotation_c2w = look_rotation(yaw, np.deg2rad(pitch))
        state = agent.get_state()
        state.position = eye
        state.rotation = quaternion_module.from_rotation_matrix(rotation_c2w)
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        rgb = _prepare_color(obs["color"])
        depth = _prepare_depth(obs["depth"], args.height_px, args.width_px)
        coverage = float((depth > 0).mean())
        if coverage < args.min_coverage or float(rgb.std()) < args.min_color_std:
            continue
        sensor = agent.get_state().sensor_states["color"]
        pose = opencv_from_opengl_c2w(quat_to_rotmat(*_quat_parts(sensor.rotation)), sensor.position)
        if abs(np.linalg.det(pose[:, :3]) - 1.0) > 1e-3:
            continue
        views.append((rgb, depth, pose.astype(np.float32)))
    if len(views) < args.frames:
        raise RuntimeError(
            f"only kept {len(views)} of {args.frames} views after {attempts} tries. "
            "A flat image usually means this habitat-sim build no longer renders Replica PTex."
        )
    return views


def publish(dest, views, intrinsics):
    staging = dest.parent / f".{dest.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    images = staging / "images"
    depths = staging / "depths"
    images.mkdir(parents=True)
    depths.mkdir()
    names = []
    try:
        for index, (rgb, depth, _pose) in enumerate(views):
            name = f"{index:05d}.jpg"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(images / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"could not write {name}")
            write_exr(depths / f"{index:05d}.exr", depth)
            names.append(name)
        (staging / "image_names.json").write_text(json.dumps(names, indent=2) + "\n")
        np.save(staging / "cam_from_worlds.npy", np.stack([pose for _rgb, _depth, pose in views]))
        np.save(staging / "intrinsics.npy", np.stack([intrinsics for _ in views]))
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


def convert_scene(scene, dest, args):
    sim, quaternion_module = make_simulator(scene, args.width_px, args.height_px, args.hfov)
    try:
        rng = np.random.default_rng(args.seed + sum(scene.name.encode()) * 997)
        views = sample_views(sim, quaternion_module, args, rng)
    finally:
        sim.close()
    publish(dest, views, pinhole(args.width_px, args.height_px, args.hfov))
    coverage = float(np.mean([(depth > 0).mean() for _rgb, depth, _pose in views]))
    return {"scene": dest.name, "source": scene.name, "frames": len(views), "depth_coverage": coverage, "status": "ok"}


def self_test():
    identity = opencv_from_opengl_c2w(np.eye(3), np.zeros(3))
    # OpenGL looks down -Z. The point in front of the camera must land at positive OpenCV z.
    front = identity[:, :3] @ np.array([0.0, 0.0, -5.0]) + identity[:, 3]
    assert np.allclose(front, [0, 0, 5]), front
    up = identity[:, :3] @ np.array([0.0, 1.0, -5.0]) + identity[:, 3]
    assert up[1] < 0
    rotation = look_rotation(0.0, 0.0)
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-6
    pose = opencv_from_opengl_c2w(rotation, np.array([1.0, 2.0, 1.5]))
    assert abs(np.linalg.det(pose[:, :3]) - 1.0) < 1e-6
    matrix = pinhole(640, 480, 90)
    assert matrix[0, 0] > 0 and matrix[0, 2] == 320 and matrix[1, 2] == 240
    print("self-test passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", nargs="?", type=Path, help="Replica root, one folder per scene")
    parser.add_argument("dst", nargs="?", type=Path, help="unified root; replica_* folders are added beside existing scenes")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--width-px", type=int, default=640)
    parser.add_argument("--height-px", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=90.0, help="horizontal field of view, degrees")
    parser.add_argument("--height", type=float, default=1.5, help="camera height above the floor, metres")
    parser.add_argument("--pitch", type=float, default=15.0, help="max random pitch, degrees")
    parser.add_argument("--min-coverage", type=float, default=0.2, help="minimum fraction of pixels with depth")
    parser.add_argument("--min-color-std", type=float, default=8.0, help="reject flat untextured renders")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
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

    try:
        import habitat_sim  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "habitat-sim is not installed. Replica here is a PTex mesh, so views have to be rendered. "
            "Install a habitat-sim build that still has the PTex shader (0.2.x) into this environment, then rerun."
        ) from exc

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
            summary = convert_scene(scene, dest, args)
        except Exception as exc:
            failed += 1
            print(f"FAIL {scene.name}: {exc}")
            report.append({"scene": dest.name, "source": scene.name, "status": "failed", "error": str(exc)})
            continue
        report.append(summary)
        print(f"ok {dest.name}: {summary['frames']} frames, depth covers {summary['depth_coverage'] * 100:.1f}%")

    (dst / "replica_convert_report.json").write_text(json.dumps(report, indent=2) + "\n")
    ok = sum(item["status"] == "ok" for item in report)
    print(f"{ok} converted, {failed} failed, {len(report) - ok - failed} skipped")
    if failed:
        return 1
    print("Next: python training/dataset_preparation/split_sequences.py", dst, "--val-per-source 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
