"""Rasterize Replica meshes into the unified training layout.

Replica ships a quad mesh with vertex colors (`mesh.ply`) plus separate PTex HDR
atlases. The atlases need ReplicaRenderer or habitat-sim. This script never
calls either. It reads the vertex colors already stored on `mesh.ply`, samples
the surface, and z-buffers those points from a walk about 1.5 m above the floor.

Replica is Z-up in the common case. The up axis is the one whose thinnest slab
contains the most vertices, which is the floor or the ceiling, and the walk
stays on the low end of that axis. Camera frames are OpenCV: x right, y down,
z forward. Depth is that z in metres, and 0 means no surface. Pixel centers
follow the loader, which unprojects `(col + 0.5, row + 0.5)`.

    uv run python training/dataset_preparation/convert_replica.py \
        ~/arda/vggt-omega/replica_data \
        ~/arda/vggt-omega/MERGED_DIR
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

WIDTH = 640
HEIGHT = 480
CAMERA_HEIGHT = 1.5
POINT_SPACING = 0.01
FRAME_COUNT = 64
MIN_FRAMES = 3

PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def intrinsics(width, height):
    """90 degree horizontal field of view. Principal point is the image center."""
    focal = width / 2.0
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def look_pose(eye, forward, up_axis):
    """Camera-from-world 4x4. `forward` is the direction the camera looks."""
    world_up = np.zeros(3, dtype=np.float64)
    world_up[up_axis] = 1.0
    forward = np.asarray(forward, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = world_up - forward * np.dot(world_up, forward)
    if np.linalg.norm(up) < 1e-8:
        up = np.zeros(3, dtype=np.float64)
        up[(up_axis + 1) % 3] = 1.0
    up /= np.linalg.norm(up)
    down = -up
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4, dtype=np.float64)
    pose[0, :3] = right
    pose[1, :3] = down
    pose[2, :3] = forward
    pose[:3, 3] = -pose[:3, :3] @ np.asarray(eye, dtype=np.float64)
    return pose


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


def _ply_endian(token):
    if "binary_little_endian" in token:
        return "<"
    if "binary_big_endian" in token:
        return ">"
    raise RuntimeError(f"unsupported ply format {token!r}; Replica mesh.ply is binary")


def _parse_ply_header(lines):
    if not lines or lines[0] != "ply":
        raise RuntimeError("not a ply file")
    endian = _ply_endian(lines[1])
    elements = []
    current = None
    for line in lines[2:]:
        parts = line.split()
        if not parts or parts[0] == "comment" or parts[0] == "obj_info":
            continue
        if parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["props"].append(("list", parts[2], parts[3], parts[4]))
            else:
                current["props"].append(("scalar", parts[1], parts[2]))
        elif parts[0] == "end_header":
            break
    return endian, elements


def _dtype(endian, name):
    base = PLY_TYPES.get(name)
    if base is None:
        raise RuntimeError(f"unsupported ply type {name}")
    if base[0] in "iuf":
        return np.dtype(endian + base)
    return np.dtype(base)


def _read_element(blob, cursor, element, endian):
    count = element["count"]
    props = element["props"]
    if any(prop[0] == "list" for prop in props):
        return _read_list_element(blob, cursor, element, endian)
    dtype = np.dtype([(prop[2], _dtype(endian, prop[1])) for prop in props])
    end = cursor + count * dtype.itemsize
    raw = np.frombuffer(bytes(blob[cursor:end]), dtype=dtype, count=count)
    return raw, end


def _read_list_element(blob, cursor, element, endian):
    """Every Replica face is a quad, so the list stride is constant.

    Extra scalar face properties, such as a texture id, are skipped.
    """
    props = element["props"]
    if not props or props[0][0] != "list" or any(prop[0] == "list" for prop in props[1:]):
        raise RuntimeError(f"{element['name']} has an unsupported property mix")
    _, count_type, index_type, name = props[0]
    count_dtype = _dtype(endian, count_type)
    index_dtype = _dtype(endian, index_type)
    trailing = sum(_dtype(endian, prop[1]).itemsize for prop in props[1:])
    if element["count"] == 0:
        return {"name": name, "indices": np.empty((0, 0), dtype=index_dtype)}, cursor
    first = int(np.frombuffer(bytes(blob[cursor : cursor + count_dtype.itemsize]), dtype=count_dtype, count=1)[0])
    index_bytes = first * index_dtype.itemsize
    stride = count_dtype.itemsize + index_bytes + trailing
    nbytes = element["count"] * stride
    raw = np.frombuffer(bytes(blob[cursor : cursor + nbytes]), dtype=np.uint8).reshape(element["count"], stride)
    counts = np.frombuffer(np.ascontiguousarray(raw[:, : count_dtype.itemsize]), dtype=count_dtype)
    if not np.all(counts == first):
        raise RuntimeError(f"{element['name']} mixes face sizes; expected every face to have {first} vertices")
    payload = np.ascontiguousarray(raw[:, count_dtype.itemsize : count_dtype.itemsize + index_bytes])
    indices = np.frombuffer(payload, dtype=index_dtype).reshape(element["count"], first)
    return {"name": name, "indices": indices}, cursor + nbytes


def _colors_from_vertices(table):
    names = {name.lower(): name for name in table.dtype.names}
    channels = []
    for options in (("red", "r", "diffuse_red"), ("green", "g", "diffuse_green"), ("blue", "b", "diffuse_blue")):
        found = next((names[option] for option in options if option in names), None)
        if found is None:
            return None
        channels.append(np.asarray(table[found]))
    rgb = np.stack(channels, axis=1)
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
        if rgb.size and float(np.nanmax(rgb)) > 1.5:
            rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    if rgb.size == 0 or float(rgb.std()) < 1e-4:
        return None
    return np.ascontiguousarray(rgb, dtype=np.float32)


def _vertex_normals(verts, tris):
    a = verts[tris[:, 0]]
    b = verts[tris[:, 1]]
    c = verts[tris[:, 2]]
    normal = np.cross(b - a, c - a).astype(np.float32)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal /= np.maximum(length, 1e-12)
    accumulated = np.zeros_like(verts)
    for column in range(3):
        np.add.at(accumulated, tris[:, column], normal)
    length = np.linalg.norm(accumulated, axis=1, keepdims=True)
    accumulated /= np.maximum(length, 1e-12)
    return accumulated


def read_mesh(path):
    with path.open("rb") as handle:
        lines = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"{path} ended inside the ply header")
            lines.append(line.decode("ascii", "replace").strip())
            if lines[-1] == "end_header":
                break
        payload = handle.read()
    endian, elements = _parse_ply_header(lines)
    cursor = 0
    vertices = None
    faces = None
    for element in elements:
        parsed, cursor = _read_element(payload, cursor, element, endian)
        if element["name"] == "vertex":
            vertices = parsed
        elif element["name"] == "face":
            faces = parsed
    if vertices is None or faces is None:
        raise RuntimeError(f"{path} is missing vertex or face data")
    names = set(vertices.dtype.names)
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise RuntimeError(f"{path} vertices have no {axis}")
    verts = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    indices = np.asarray(faces["indices"])
    if indices.shape[1] == 4:
        tris = np.empty((indices.shape[0] * 2, 3), dtype=np.int64)
        tris[0::2] = indices[:, [0, 1, 2]]
        tris[1::2] = indices[:, [0, 2, 3]]
    elif indices.shape[1] == 3:
        tris = indices.astype(np.int64, copy=False)
    else:
        raise RuntimeError(f"{path} faces have {indices.shape[1]} vertices")
    if tris.size and (int(tris.min()) < 0 or int(tris.max()) >= len(verts)):
        raise RuntimeError(f"{path} faces index outside the vertex list")
    colors = _colors_from_vertices(vertices)
    source = "vertex"
    if colors is None:
        source = "normal"
        colors = np.clip(0.5 + 0.5 * _vertex_normals(verts, tris), 0.0, 1.0).astype(np.float32)
    return verts, tris, colors, source


def _unique_rows(keys):
    keys = np.ascontiguousarray(keys)
    packed = keys.view(np.dtype((np.void, keys.shape[1] * keys.dtype.itemsize))).ravel()
    _, index = np.unique(packed, return_index=True)
    return index


def voxel_downsample(points, colors, spacing):
    if len(points) == 0:
        return points.astype(np.float32), colors.astype(np.float32)
    keys = np.floor(np.asarray(points, dtype=np.float64) / spacing).astype(np.int32)
    index = _unique_rows(keys)
    return np.ascontiguousarray(points[index], dtype=np.float32), np.ascontiguousarray(colors[index], dtype=np.float32)


def _edge_samples(origin, dest, origin_color, dest_color, spacing):
    length = np.linalg.norm(dest - origin, axis=1)
    # Long edges are rare on Replica's dense scan. A coarse face still needs
    # samples about `spacing` apart, so only a handful of long edges may use a
    # finer cap. A dense mesh stays on the cheap cap.
    cap = 80 if int(np.count_nonzero(length > 2.0 * spacing)) < 50_000 else 16
    steps = np.clip(np.ceil(length / spacing).astype(np.int32), 1, cap)
    if len(steps) == 0 or int(steps.max()) <= 1:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    points = []
    colors = []
    for step in range(1, int(steps.max())):
        mask = steps > step
        if not np.any(mask):
            continue
        weight = (step / steps[mask]).astype(np.float32)[:, None]
        points.append(origin[mask] * (1.0 - weight) + dest[mask] * weight)
        colors.append(origin_color[mask] * (1.0 - weight) + dest_color[mask] * weight)
    return np.concatenate(points), np.concatenate(colors)


def _interior_samples(a, b, c, ca, cb, cc, spacing):
    """Fill faces much larger than `spacing`. Dense Replica faces skip this."""
    longest = np.maximum(
        np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1)),
        np.linalg.norm(c - b, axis=1),
    )
    mask = longest > 2.0 * spacing
    if not np.any(mask):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    a, b, c = a[mask], b[mask], c[mask]
    ca, cb, cc = ca[mask], cb[mask], cc[mask]
    steps = np.clip(np.ceil(longest[mask] / spacing).astype(np.int32), 3, 8 if len(a) > 20_000 else 40)
    grid = int(steps.max())
    points = []
    colors = []
    for i in range(1, grid):
        for j in range(1, grid - i):
            use = steps > (i + j)
            if not np.any(use):
                continue
            u = (i / steps[use]).astype(np.float32)[:, None]
            v = (j / steps[use]).astype(np.float32)[:, None]
            w = 1.0 - u - v
            points.append(u * a[use] + v * b[use] + w * c[use])
            colors.append(u * ca[use] + v * cb[use] + w * cc[use])
    if not points:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    return np.concatenate(points).astype(np.float32), np.concatenate(colors).astype(np.float32)


def surface_cloud(verts, tris, colors, spacing):
    """Points about `spacing` metres apart, carrying interpolated vertex color."""
    reduced = [voxel_downsample(verts, colors, spacing)]
    batch = 400_000
    for start in range(0, len(tris), batch):
        face = tris[start : start + batch]
        a, b, c = verts[face[:, 0]], verts[face[:, 1]], verts[face[:, 2]]
        ca, cb, cc = colors[face[:, 0]], colors[face[:, 1]], colors[face[:, 2]]
        samples = [
            (a + b + c) / 3.0,
            (2.0 * a + b + c) / 4.0,
            (a + 2.0 * b + c) / 4.0,
            (a + b + 2.0 * c) / 4.0,
        ]
        sample_colors = [
            (ca + cb + cc) / 3.0,
            (2.0 * ca + cb + cc) / 4.0,
            (ca + 2.0 * cb + cc) / 4.0,
            (ca + cb + 2.0 * cc) / 4.0,
        ]
        for origin, dest, origin_color, dest_color in ((a, b, ca, cb), (b, c, cb, cc), (c, a, cc, ca)):
            extra, extra_color = _edge_samples(origin, dest, origin_color, dest_color, spacing)
            if len(extra):
                samples.append(extra)
                sample_colors.append(extra_color)
        extra, extra_color = _interior_samples(a, b, c, ca, cb, cc, spacing)
        if len(extra):
            samples.append(extra)
            sample_colors.append(extra_color)
        reduced.append(voxel_downsample(np.concatenate(samples), np.concatenate(sample_colors), spacing))
    points = np.concatenate([item[0] for item in reduced])
    point_colors = np.concatenate([item[1] for item in reduced])
    return voxel_downsample(points, point_colors, spacing)


def detect_up_axis(points):
    """Axis perpendicular to the biggest flat surface. Ties prefer Z, Replica's up axis."""
    best_axis = 2
    best_count = -1
    for axis in (2, 1, 0):
        values = points[:, axis]
        low = float(values.min())
        high = float(values.max())
        if high - low < 1e-6:
            continue
        bins = np.floor((values - low) / 0.05).astype(np.int32)
        count = int(np.bincount(bins).max())
        if count > best_count:
            best_count = count
            best_axis = axis
    return best_axis


def _order_xy(xy):
    count = len(xy)
    if count <= 1:
        return xy
    used = np.zeros(count, dtype=bool)
    current = int(np.lexsort((xy[:, 1], xy[:, 0]))[0])
    order = np.empty(count, dtype=np.int64)
    for index in range(count):
        order[index] = current
        used[current] = True
        if index == count - 1:
            break
        distance = np.linalg.norm(xy - xy[current], axis=1)
        distance[used] = np.inf
        current = int(np.argmin(distance))
    return xy[order]


def floor_waypoints(verts, up_axis):
    height = verts[:, up_axis]
    floor_h = float(np.percentile(height, 2))
    band = np.abs(height - floor_h) <= 0.1
    if int(band.sum()) < 30:
        band = np.abs(height - floor_h) <= 0.3
    horizontal = [axis for axis in range(3) if axis != up_axis]
    xy = verts[band][:, horizontal]
    if len(xy) == 0:
        raise RuntimeError("no vertices near the floor")
    voxel = 0.6
    index = _unique_rows(np.floor(xy / voxel).astype(np.int32))
    xy = xy[index]
    if len(xy) > 80:
        index = _unique_rows(np.floor(xy / 1.2).astype(np.int32))
        xy = xy[index]
    return _order_xy(xy.astype(np.float64)), floor_h, horizontal


def _eye_from_xy(xy, height, up_axis, horizontal):
    eye = np.zeros(3, dtype=np.float64)
    eye[horizontal[0]] = xy[0]
    eye[horizontal[1]] = xy[1]
    eye[up_axis] = height
    return eye


def _poses_along(xy, floor_h, up_axis, horizontal, count):
    if len(xy) == 1:
        return []
    delta = np.diff(xy, axis=0)
    segment = np.linalg.norm(delta, axis=1)
    total = float(segment.sum())
    if total < 0.3:
        return []
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    height = floor_h + CAMERA_HEIGHT
    poses = []
    for distance in np.linspace(0.0, total, count, endpoint=False):
        segment_index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment_index = min(max(segment_index, 0), len(segment) - 1)
        local = (distance - cumulative[segment_index]) / max(float(segment[segment_index]), 1e-6)
        local = min(max(local, 0.0), 1.0)
        point = xy[segment_index] * (1.0 - local) + xy[segment_index + 1] * local
        direction = delta[segment_index]
        yaw = math.atan2(float(direction[1]), float(direction[0])) + 0.15 * math.sin(distance)
        forward = np.zeros(3, dtype=np.float64)
        forward[horizontal[0]] = math.cos(yaw)
        forward[horizontal[1]] = math.sin(yaw)
        poses.append(look_pose(_eye_from_xy(point, height, up_axis, horizontal), forward, up_axis)[:3])
    return poses


def _turntable(xy, floor_h, up_axis, horizontal, count):
    center = xy.mean(axis=0)
    spread = np.linalg.norm(xy - center, axis=1)
    radius = max(float(np.median(spread)) * 0.45, 0.4)
    height = floor_h + CAMERA_HEIGHT
    poses = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        point = center + radius * np.array([math.cos(angle), math.sin(angle)])
        eye = _eye_from_xy(point, height, up_axis, horizontal)
        forward = _eye_from_xy(center, height, up_axis, horizontal) - eye
        forward[up_axis] = 0.0
        if np.linalg.norm(forward) < 1e-8:
            forward = np.zeros(3, dtype=np.float64)
            forward[horizontal[0]] = 1.0
        poses.append(look_pose(eye, forward, up_axis)[:3])
    return poses


def trajectory(verts, up_axis, count):
    """A walk along the floor, plus a ring of views used only if the walk is blocked."""
    xy, floor_h, horizontal = floor_waypoints(verts, up_axis)
    path = _poses_along(xy, floor_h, up_axis, horizontal, count)
    spin = _turntable(xy, floor_h, up_axis, horizontal, count)
    if len(path) >= MIN_FRAMES:
        return path, spin
    return spin, []


def render_points(points, colors, pose, matrix, height, width):
    rotation = np.asarray(pose[:, :3], dtype=np.float32)
    translation = np.asarray(pose[:, 3], dtype=np.float32)
    camera = points @ rotation.T + translation
    keep = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.05)
    camera = camera[keep]
    if len(camera) == 0:
        return np.zeros((height, width, 3), np.float32), np.zeros((height, width), np.float32)
    depth_z = camera[:, 2]
    columns = matrix[0, 0] * (camera[:, 0] / depth_z) + matrix[0, 2]
    rows = matrix[1, 1] * (camera[:, 1] / depth_z) + matrix[1, 2]
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height) & np.isfinite(columns) & np.isfinite(rows)
    columns = np.floor(columns[inside]).astype(np.int32)
    rows = np.floor(rows[inside]).astype(np.int32)
    depth_z = depth_z[inside]
    rgb = colors[keep][inside]
    image = np.zeros((height, width, 3), np.float32)
    depth = np.zeros((height, width), np.float32)
    if len(depth_z) == 0:
        return image, depth
    pixels = rows.astype(np.int64) * width + columns
    order = np.argsort(depth_z)
    _, first = np.unique(pixels[order], return_index=True)
    chosen = order[first]
    depth.ravel()[pixels[chosen]] = depth_z[chosen]
    image.reshape(-1, 3)[pixels[chosen]] = rgb[chosen]
    return image, depth


def fill_small_holes(depth, color, radius):
    filled_depth = depth.copy()
    filled_color = color.copy()
    for _ in range(radius):
        acc_depth = np.zeros_like(filled_depth)
        acc_color = np.zeros_like(filled_color)
        acc_count = np.zeros(filled_depth.shape, np.float32)
        for shift in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            shifted_depth = np.roll(filled_depth, shift, axis=(0, 1))
            shifted_color = np.roll(filled_color, shift, axis=(0, 1))
            mask = np.roll(filled_depth > 0, shift, axis=(0, 1))
            if shift[0] == 1:
                mask[0, :] = False
            elif shift[0] == -1:
                mask[-1, :] = False
            if shift[1] == 1:
                mask[:, 0] = False
            elif shift[1] == -1:
                mask[:, -1] = False
            acc_depth += shifted_depth * mask
            acc_color += shifted_color * mask[:, :, None]
            acc_count += mask
        take = (depth == 0) & (filled_depth == 0) & (acc_count > 0)
        filled_depth[take] = acc_depth[take] / acc_count[take]
        filled_color[take] = acc_color[take] / acc_count[take, None]
    return filled_depth, filled_color


def _acceptable(depth):
    valid = depth > 0
    coverage = float(valid.mean())
    if coverage < 0.2:
        return False, coverage
    median = float(np.median(depth[valid]))
    return median >= 0.4, coverage


def render_scene(points, colors, poses, matrix, desc):
    kept = []
    coverages = []
    for pose in tqdm(poses, desc=desc, leave=False):
        image, depth = render_points(points, colors, pose, matrix, HEIGHT, WIDTH)
        depth, image = fill_small_holes(depth, image, radius=2)
        ok, coverage = _acceptable(depth)
        coverages.append(coverage)
        if ok:
            kept.append((pose.astype(np.float32), image, depth, coverage))
    return kept, coverages


def publish(dest, frames):
    staging = dest.parent / f".{dest.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    images = staging / "images"
    depths = staging / "depths"
    images.mkdir(parents=True)
    depths.mkdir()
    names = []
    poses = []
    coverages = []
    matrix = intrinsics(WIDTH, HEIGHT)
    try:
        for index, (pose, image, depth, coverage) in enumerate(frames):
            name = f"{index:05d}.jpg"
            bgr = np.clip(image[:, :, ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
            if not cv2.imwrite(str(images / name), bgr):
                raise RuntimeError(f"failed to write {name}")
            write_exr(depths / f"{index:05d}.exr", depth.astype(np.float32))
            names.append(name)
            poses.append(pose)
            coverages.append(coverage)
        (staging / "image_names.json").write_text(json.dumps(names, indent=2) + "\n")
        np.save(staging / "cam_from_worlds.npy", np.stack(poses).astype(np.float32))
        np.save(staging / "intrinsics.npy", np.repeat(matrix[None], len(poses), axis=0))
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
    return float(np.mean(coverages))


def self_test():
    pose = look_pose(np.array([0.0, 0.0, 1.5]), np.array([1.0, 0.0, 0.0]), up_axis=2)
    assert abs(np.linalg.det(pose[:3, :3]) - 1.0) < 1e-6
    ahead = pose[:3, :3] @ np.array([2.0, 0.0, 1.5]) + pose[:3, 3]
    assert np.allclose(ahead, [0, 0, 2], atol=1e-6), ahead
    right_point = pose[:3, :3] @ np.array([2.0, -1.0, 1.5]) + pose[:3, 3]
    assert right_point[0] > 0.5 and abs(right_point[2] - 2) < 1e-6

    width, height = 32, 24
    matrix = intrinsics(width, height)
    identity = np.eye(4, dtype=np.float32)[:3]
    points = np.array([[0, 0, 4], [0, 0, 2], [1, 0, 4]], np.float32)
    colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float32)
    image, depth = render_points(points, colors, identity, matrix, height, width)
    assert depth[12, 16] == 2.0, depth[12, 16]
    assert np.allclose(image[12, 16], [0, 0, 1])
    assert depth[12, 20] == 4.0
    assert np.allclose(image[12, 20], [0, 1, 0])
    center = 16.5
    unprojected = (center - matrix[0, 2]) * depth[12, 16] / matrix[0, 0]
    assert abs(float(unprojected)) < 0.5 / float(matrix[0, 0]) * 2 + 1e-6

    rng = np.random.default_rng(0)
    floor = np.zeros((400, 3), np.float32)
    floor[:, 0] = rng.uniform(0, 6, 400)
    floor[:, 1] = rng.uniform(0, 5, 400)
    walls = rng.uniform(0, 5, size=(40, 3)).astype(np.float32)
    walls[:, 2] = rng.uniform(0, 2.5, 40)
    assert detect_up_axis(np.concatenate([floor, walls])) == 2

    hole = np.zeros((5, 5), np.float32)
    hole_color = np.zeros((5, 5, 3), np.float32)
    hole[2, 2] = 3
    hole_color[2, 2] = (1, 0, 0)
    filled, filled_color = fill_small_holes(hole, hole_color, 1)
    assert filled[2, 3] == 3 and filled[1, 1] == 0
    assert np.allclose(filled_color[2, 3], [1, 0, 0])

    import tempfile

    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    # Quad diagonal matches read_mesh: (0, 1, 2) and (0, 2, 3).
    quads = np.array([[0, 1, 2, 3]], np.int32)
    vertex_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], np.uint8)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(verts)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {len(quads)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    table = np.empty(len(verts), dtype=vertex_dtype)
    table["x"], table["y"], table["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    table["red"], table["green"], table["blue"] = vertex_colors[:, 0], vertex_colors[:, 1], vertex_colors[:, 2]
    face_rows = []
    for face in quads:
        face_rows.append(bytes([4]) + np.asarray(face, dtype="<i4").tobytes())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mesh.ply"
        path.write_bytes(header + table.tobytes() + b"".join(face_rows))
        loaded, tris, colors, source = read_mesh(path)
        assert source == "vertex"
        assert np.allclose(loaded, verts)
        assert len(tris) == 2
        assert colors.shape == (4, 3) and colors[0, 0] > 0.9 and colors[1, 1] > 0.9
        depth = np.array([[0, 1.5], [0, 2.5]], np.float32)
        write_exr(Path(tmp) / "d.exr", depth)
        restored = cv2.imread(str(Path(tmp) / "d.exr"), cv2.IMREAD_UNCHANGED)
        assert np.allclose(restored, depth)
    cloud, cloud_colors = surface_cloud(verts, tris, colors, 0.4)
    assert len(cloud) >= 4
    assert cloud_colors.shape == (len(cloud), 3)
    print("self-test passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", nargs="?", type=Path, help="Replica root, one folder per scene")
    parser.add_argument("dst", nargs="?", type=Path, help="unified root; replica_* folders are added beside existing scenes")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.src is None or args.dst is None:
        parser.error("src and dst are required")
    if args.frames < MIN_FRAMES:
        parser.error(f"--frames must be at least {MIN_FRAMES}")

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

    matrix = intrinsics(WIDTH, HEIGHT)
    report = []
    failed = 0
    for scene in scenes:
        dest = dst / output_name(scene.name)
        if already_done(dest, args.overwrite):
            print(f"skip {dest.name}", flush=True)
            report.append({"scene": dest.name, "source": scene.name, "status": "skipped"})
            continue
        started = time.time()
        print(f"loading {scene.name}", flush=True)
        try:
            verts, tris, colors, source = read_mesh(scene / "mesh.ply")
            up_axis = detect_up_axis(verts)
            points, point_colors = surface_cloud(verts, tris, colors, POINT_SPACING)
            poses, backup = trajectory(verts, up_axis, args.frames)
            print(
                f"{scene.name}: {len(points) / 1e6:.2f}M points, up={'xyz'[up_axis]}, color={source}, {len(poses)} cameras",
                flush=True,
            )
            frames, coverages = render_scene(points, point_colors, poses, matrix, scene.name)
            if len(frames) < MIN_FRAMES and backup:
                print(f"{scene.name}: walk was blocked, rendering a ring of views", flush=True)
                frames, coverages = render_scene(points, point_colors, backup, matrix, scene.name + " ring")
            if len(frames) < MIN_FRAMES:
                best = max(coverages) if coverages else 0.0
                raise RuntimeError(f"only {len(frames)} usable frames (best coverage {best * 100:.1f}%)")
            coverage = publish(dest, frames)
        except Exception as exc:
            failed += 1
            print(f"FAIL {scene.name}: {exc}", flush=True)
            report.append({"scene": dest.name, "source": scene.name, "status": "failed", "error": str(exc)})
            continue
        report.append(
            {
                "scene": dest.name,
                "source": scene.name,
                "frames": len(frames),
                "depth_coverage": coverage,
                "color": source,
                "up_axis": "xyz"[up_axis],
                "width": WIDTH,
                "height": HEIGHT,
                "seconds": round(time.time() - started, 1),
                "status": "ok",
            }
        )
        print(
            f"ok {dest.name}: {len(frames)} frames, depth covers {coverage * 100:.1f}%, {time.time() - started:.0f}s",
            flush=True,
        )

    (dst / "replica_convert_report.json").write_text(json.dumps(report, indent=2) + "\n")
    ok = sum(item["status"] == "ok" for item in report)
    print(f"{ok} converted, {failed} failed, {len(report) - ok - failed} skipped")
    if failed:
        return 1
    print("Next: python training/dataset_preparation/split_sequences.py", dst, "--val-per-source 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
