"""Panoramas from ordinary photos, through 3D geometry instead of 2D homographies.

1. VGGT-Omega predicts a camera pose, a depth map and a confidence map for every photo.
2. A virtual camera is placed in the scene and the canvas is its image. Two kinds:
   - perspective (default): a pinhole camera at the mean camera centre, aimed at the middle of
     the photos. Straight lines stay straight. Right for a camera that turns on the spot.
   - plane: an orthographic camera looking straight at the dominant plane of the scene (a shelf,
     a facade, a wall). Every canvas pixel is the same size on the plane, so nothing stretches
     towards the edges. Right for a camera that walks along the scene.
3. A z-buffer built from all depth maps gives, for every canvas pixel, the surface that is
   closest to the virtual camera. Parallax is handled here: a pole in front of a building
   wins over the building, whichever photo saw it.
4. Every canvas pixel is then looked up in each photo (backward warping, so the result is
   sharp). A photo contributes only if the surface it sees there lies at the same depth.
   Contributions are blended with confidence and image-border weights, after a per-photo
   exposure gain is estimated from the overlaps.
5. Sky pixels have no usable depth. They are placed at a very large distance, which turns
   them into an ordinary rotation-only panorama.

Colour comes from the original photos. Only pose, depth and confidence come from the
model, which runs at a low resolution, so the panorama can be far sharper than the model input.
"""

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

CENTERS = ("mean", "median", "first")
BLENDS = ("multiband", "average")
PROJECTIONS = ("perspective", "plane")
SKY_THRESHOLD = 32  # skyseg output below this value is not sky
_MIN_AXIS = 0.1  # cosine of the angle to the canvas axis below which a direction is off the canvas
_MULTIBAND_LIMIT = 150_000_000  # photos x canvas pixels above which multiband falls back to the average


@dataclass
class Settings:
    projection: str = "perspective"  # perspective: camera that turns. plane: camera that walks along a plane
    scale: float = 1.0  # 1.0 matches the detail of the photos
    center: str = "mean"
    blend: str = "multiband"  # multiband: best photo per pixel, joined by pyramid blending. average: weighted mean
    fill_gaps: bool = True  # where no photo passes the strict tests, fall back to the best photo that sees the pixel
    conf_percentile: float = 15.0  # drop this share of the least confident pixels per photo
    depth_tolerance: float = 0.08  # relative depth disagreement that still counts as the same surface
    feather: float = 0.2  # border ramp, as a fraction of the shorter image side
    edge_rtol: float = 0.05  # drop pixels whose 3x3 depth range exceeds this share of the depth
    exposure: bool = True
    use_sky: bool = True
    max_megapixels: float = 50.0


@dataclass
class Reconstruction:
    colors: torch.Tensor  # (N, 3, H, W) in [0, 1], at colour resolution
    depth: torch.Tensor  # (N, H, W)
    conf: torch.Tensor  # (N, H, W), at least 1
    edge: torch.Tensor  # (N, H, W) bool, flying pixels at depth discontinuities
    sky: torch.Tensor | None  # (N, H, W) bool
    extrinsic: torch.Tensor  # (N, 3, 4) camera-from-world
    intrinsic: torch.Tensor  # (N, 3, 3) at colour resolution
    timings: dict = field(default_factory=dict)


@dataclass
class Panorama:
    image: np.ndarray  # (H, W, 3) uint8
    alpha: np.ndarray  # (H, W) uint8, 255 where the photos cover the canvas
    views: np.ndarray  # (H, W) uint8, number of photos blended per pixel
    depth: np.ndarray  # (H, W) float32, distance to the virtual camera, NaN at sky and holes
    info: dict
    layout: "Layout" = None


# ---------------------------------------------------------------- model side


def load_model(checkpoint: str, device: torch.device):
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(autocast=device.type == "cuda").eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device)


def load_sky_session(path: str):
    """ONNX sky segmenter, or None when onnxruntime or the file is missing."""
    try:
        import onnxruntime
    except ImportError:
        return None
    try:
        return onnxruntime.InferenceSession(path)
    except Exception:
        return None


def _segment_sky(session, rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from visual_util import run_skyseg

    result = run_skyseg(session, [320, 320], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    result = cv2.resize(result, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return result >= SKY_THRESHOLD


def depth_edges(depth: torch.Tensor, rtol: float) -> torch.Tensor:
    """Pixels whose 3x3 neighbourhood spans more than `rtol` of their depth."""
    value = depth[:, None]
    spread = F.max_pool2d(value, 3, 1, 1) + F.max_pool2d(-value, 3, 1, 1)
    return (spread / value.clamp_min(1e-6) > rtol)[:, 0]


@torch.inference_mode()
def reconstruct(
    model,
    paths: list[str],
    device: torch.device,
    resolution: int = 512,
    color_long_side: int = 1600,
    sky_session=None,
    edge_rtol: float = 0.05,
) -> Reconstruction:
    """Run the model on the photos and resample its output to the colour resolution."""
    from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio
    from vggt_omega.utils.pose_enc import encoding_to_camera

    started = time.time()
    photos = []
    for path in paths:
        with Image.open(path) as image:
            photos.append(_crop_to_supported_aspect_ratio(ImageOps.exif_transpose(image).convert("RGB")))
    shapes = {_balanced_target_shape(photo.height / photo.width, resolution, 16) for photo in photos}
    if len(shapes) != 1:
        raise ValueError("All photos need the same size and orientation. Mixed portrait and landscape is not supported.")
    model_height, model_width = shapes.pop()
    first = photos[0]
    shrink = min(1.0, color_long_side / max(first.size))
    color_width, color_height = round(first.width * shrink), round(first.height * shrink)

    def to_tensor(image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

    model_images = [photo.resize((model_width, model_height), Image.Resampling.BICUBIC) for photo in photos]
    colors = torch.stack([to_tensor(photo.resize((color_width, color_height), Image.Resampling.LANCZOS)) for photo in photos])

    prediction = model(torch.stack([to_tensor(image) for image in model_images]).to(device))
    depth = prediction["depth"][0].float().cpu()
    conf = prediction["depth_conf"][0].float().cpu()
    depth = depth[..., 0] if depth.ndim == 4 else depth
    conf = conf[..., 0] if conf.ndim == 4 else conf
    extrinsic, intrinsic = encoding_to_camera(prediction["pose_enc"], (model_height, model_width))
    extrinsic, intrinsic = extrinsic[0].float().cpu(), intrinsic[0].float().cpu()

    edge = depth_edges(depth, edge_rtol)
    sky = None
    if sky_session is not None:
        sky = torch.from_numpy(
            np.stack([_segment_sky(sky_session, np.asarray(image), (model_height, model_width)) for image in model_images])
        )

    size = (color_height, color_width)

    def resample(value: torch.Tensor, mode: str) -> torch.Tensor:
        return F.interpolate(value[:, None].float(), size=size, mode=mode)[:, 0]

    intrinsic = intrinsic.clone()
    intrinsic[:, 0] *= color_width / model_width
    intrinsic[:, 1] *= color_height / model_height
    reconstruction = Reconstruction(
        colors=colors,
        depth=resample(depth, "bilinear"),
        conf=resample(conf, "bilinear"),
        edge=resample(edge, "nearest") > 0.5,
        sky=None if sky is None else resample(sky, "nearest") > 0.5,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
    )
    reconstruction.timings["reconstruct"] = time.time() - started
    return reconstruction


# ---------------------------------------------------------------- geometry


def camera_centers(extrinsic: torch.Tensor) -> torch.Tensor:
    return -torch.einsum("nji,nj->ni", extrinsic[:, :, :3], extrinsic[:, :, 3])


def choose_center(centers: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "first":
        return centers[0]
    if mode == "median":
        return centers.median(dim=0).values
    return centers.mean(dim=0)


def pano_rotation(extrinsic: torch.Tensor) -> torch.Tensor:
    """World-to-panorama rotation: y is down, z is the first camera's heading, x is to the right."""
    rotation = extrinsic[:, :, :3]
    up = F.normalize(-rotation[:, 1, :].mean(dim=0), dim=0)
    heading = rotation[0, 2, :]
    heading = heading - (heading @ up) * up
    if heading.norm() < 1e-3:
        heading = rotation[0, 0, :] - (rotation[0, 0, :] @ up) * up
    z = F.normalize(heading, dim=0)
    y = -up
    return torch.stack([torch.linalg.cross(y, z), y, z])


def pixel_rays(intrinsic: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Rays with z = 1 through the pixel centres, flattened to (H * W, 3)."""
    ys, xs = torch.meshgrid(torch.arange(height) + 0.5, torch.arange(width) + 0.5, indexing="ij")
    x = (xs - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (ys - intrinsic[1, 2]) / intrinsic[1, 1]
    return torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)


class Points(NamedTuple):
    world: torch.Tensor  # (P, 3) world positions. Sky is a point `far` away along its ray.
    size: torch.Tensor  # (P,) world size of one source pixel at that point
    is_sky: torch.Tensor  # (P,)


def view_points(
    rec: Reconstruction,
    index: int,
    center: torch.Tensor,
    far: float,
    threshold: float,
    use_sky: bool,
    stride: int = 1,
) -> Points:
    """Usable pixels of one photo as world points."""
    height, width = rec.depth.shape[-2:]
    rotation, translation = rec.extrinsic[index, :, :3], rec.extrinsic[index, :, 3]
    depth = rec.depth[index].reshape(-1)
    sky = rec.sky[index].reshape(-1) if use_sky and rec.sky is not None else torch.zeros_like(depth, dtype=torch.bool)
    solid = torch.isfinite(depth) & (depth > 0) & ~rec.edge[index].reshape(-1) & (rec.conf[index].reshape(-1) >= threshold)
    flat = torch.nonzero((solid & ~sky) | sky).squeeze(1)[::stride]

    ray = pixel_rays(rec.intrinsic[index], height, width)[flat]
    z = depth[flat]
    is_sky = sky[flat]
    focal = 0.5 * (rec.intrinsic[index, 0, 0] + rec.intrinsic[index, 1, 1])

    world = (ray * z[:, None] - translation) @ rotation
    world = torch.where(is_sky[:, None], center + F.normalize(ray @ rotation, dim=1) * far, world)
    size = torch.where(is_sky, torch.full_like(z, far) / focal, z / focal)
    return Points(world, size, is_sky)


def confidence_thresholds(rec: Reconstruction, percentile: float) -> torch.Tensor:
    values = []
    for conf in rec.conf:
        sample = conf.reshape(-1)[::7]
        values.append(max(float(torch.quantile(sample, percentile / 100.0)), 1.0))
    return torch.tensor(values)


class _Canvas:
    """Size of a canvas whose extent is a_min..a_max by y_min..y_max in canvas units."""

    @property
    def width(self) -> int:
        return max(1, math.ceil((self.a_max - self.a_min) * self.ppr))

    @property
    def height(self) -> int:
        return max(1, math.ceil((self.y_max - self.y_min) * self.ppr))

    def with_ppr(self, ppr: float):
        return dataclasses.replace(self, ppr=ppr)


@dataclass
class Layout(_Canvas):
    """Perspective canvas: a pinhole camera at the virtual centre.

    `a` is x / z and `y` is -y / z in the panorama frame (up is positive), so every straight line
    in the scene stays straight. `ppr` is the focal length in pixels.
    """

    rotation: torch.Tensor
    a_min: float
    a_max: float
    y_min: float
    y_max: float
    ppr: float
    center: torch.Tensor | None = None

    def angles(self, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Canvas coordinates (a, y) of world directions. Directions behind the canvas get 1e6."""
        pano = direction @ self.rotation.T
        valid = pano[..., 2] > _MIN_AXIS
        z = pano[..., 2].clamp_min(_MIN_AXIS)
        off = torch.full_like(z, 1e6)
        return torch.where(valid, pano[..., 0] / z, off), torch.where(valid, -pano[..., 1] / z, off)

    def to_canvas(self, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, y = self.angles(direction)
        return (a - self.a_min) * self.ppr, (self.y_max - y) * self.ppr

    def to_directions(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a = self.a_min + u / self.ppr
        y = self.y_max - v / self.ppr
        pano = F.normalize(torch.stack([a, -y, torch.ones_like(a)], dim=-1), dim=-1)
        return pano @ self.rotation

    def project(self, world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Canvas pixel (u, v) and distance to the virtual camera of world points."""
        offset = world - self.center
        distance = offset.norm(dim=-1).clamp_min(1e-6)
        u, v = self.to_canvas(offset / distance[..., None])
        return u, v, distance

    def unproject(self, u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.center + self.to_directions(u, v) * depth[..., None]

    def splat_pixels(self, points: Points) -> torch.Tensor:
        """Canvas pixels covered by one source pixel. It grows towards the edges (1 / cos^2)."""
        offset = points.world - self.center
        distance = offset.norm(dim=-1).clamp_min(1e-6)
        axis = ((offset / distance[..., None]) @ self.rotation.T)[..., 2].clamp_min(_MIN_AXIS)
        return self.ppr * points.size / distance / axis**2

    def describe(self) -> dict:
        return {"horizontal_fov_deg": round(math.degrees(self.longitude_span()), 1)}

    def longitude_span(self) -> float:
        """Horizontal angle covered by the canvas, in radians."""
        return math.atan(self.a_max) - math.atan(self.a_min)


@dataclass
class PlaneLayout(_Canvas):
    """Orthographic canvas looking straight at a plane.

    `a` runs along `right` and `y` along `up`, both in the plane, in world units. `forward` points
    into the scene and the depth of a point is its distance from the plane along it, so a pixel
    covers the same patch of the plane everywhere and nothing stretches towards the edges.
    """

    origin: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    forward: torch.Tensor
    a_min: float
    a_max: float
    y_min: float
    y_max: float
    ppr: float  # pixels per world unit
    stats: dict = field(default_factory=dict)

    def project(self, world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offset = world - self.origin
        return (offset @ self.right - self.a_min) * self.ppr, (self.y_max - offset @ self.up) * self.ppr, offset @ self.forward

    def unproject(self, u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        a = (self.a_min + u / self.ppr)[..., None]
        y = (self.y_max - v / self.ppr)[..., None]
        return self.origin + a * self.right + y * self.up + depth[..., None] * self.forward

    def splat_pixels(self, points: Points) -> torch.Tensor:
        return self.ppr * points.size

    def describe(self) -> dict:
        return {"plane_extent_units": f"{self.a_max - self.a_min:.2f} x {self.y_max - self.y_min:.2f}", **self.stats}


def turned(rotation: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotation that also turns the panorama about its vertical axis, so longitude shifts by -angle."""
    c, s = math.cos(angle), math.sin(angle)
    turn = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=rotation.dtype)
    return turn @ rotation


def _crop(a: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float, float]:
    """Canvas extent that holds 99.8% of the points, with a small margin."""
    pick = torch.randperm(len(a))[:300000]
    quantiles = torch.tensor([0.001, 0.999])
    qa, qy = torch.quantile(a[pick], quantiles), torch.quantile(y[pick], quantiles)
    pad_a, pad_y = 0.01 * float(qa[1] - qa[0]), 0.01 * float(qy[1] - qy[0])
    return float(qa[0]) - pad_a, float(qa[1]) + pad_a, float(qy[0]) - pad_y, float(qy[1]) + pad_y


def _cap_resolution(layout, settings: Settings):
    megapixels = layout.width * layout.height / 1e6
    if megapixels > settings.max_megapixels:
        layout = layout.with_ppr(layout.ppr * math.sqrt(settings.max_megapixels / megapixels))
    return layout


def plan_perspective(rec: Reconstruction, center: torch.Tensor, thresholds: torch.Tensor, far: float, settings: Settings) -> Layout:
    """Aim a pinhole camera at the middle of the photos and crop the canvas to what they cover."""
    rotation = pano_rotation(rec.extrinsic)
    stride = max(1, rec.depth[0].numel() // 40000)
    samples = F.normalize(
        torch.cat([view_points(rec, i, center, far, float(thresholds[i]), settings.use_sky, stride).world for i in range(len(rec.depth))])
        - center,
        dim=1,
    )
    pano = samples @ rotation.T
    lon = torch.atan2(pano[:, 0], pano[:, 2])
    rotation = turned(rotation, float(torch.atan2(torch.sin(lon).mean(), torch.cos(lon).mean())))

    a, y = Layout(rotation, -1.0, 1.0, -1.0, 1.0, 1.0).angles(samples)
    ahead = (a.abs() < 1e5) & (y.abs() < 1e5)
    if not ahead.any():
        raise ValueError("The photos do not share a viewing direction. Perspective needs less than about 170 degrees of view.")
    a_min, a_max, y_min, y_max = _crop(a[ahead], y[ahead])
    focal = float(rec.intrinsic[:, [0, 1], [0, 1]].mean())
    layout = Layout(rotation, a_min - 0.01, a_max + 0.01, y_min - 0.01, y_max + 0.01, settings.scale * focal, center)
    return _cap_resolution(layout, settings)


def fit_plane(
    points: torch.Tensor,
    tolerance: float,
    facing: torch.Tensor,
    iterations: int = 400,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RANSAC plane through the densest sheet of points that faces the cameras.

    Planes whose normal is more than 60 degrees from `facing` (the mean viewing direction) are
    skipped, so a floor or ceiling cannot win over the wall in front. Returns a point on the plane,
    its unit normal, and a bool mask of the points within `tolerance` of it.
    """
    generator = torch.Generator().manual_seed(seed)
    sample = points[torch.randperm(len(points), generator=generator)[:30000]]
    best_count, best = -1, None
    for start in range(0, iterations, 50):
        picks = torch.randint(0, len(sample), (50, 3), generator=generator)
        origin = sample[picks[:, 0]]
        normal = torch.linalg.cross(sample[picks[:, 1]] - origin, sample[picks[:, 2]] - origin)
        length = normal.norm(dim=1)
        normal = normal / length.clamp_min(1e-12)[:, None]
        usable = (length > 1e-9) & ((normal @ facing).abs() > 0.5)
        distance = ((sample[None] - origin[:, None]) * normal[:, None]).sum(-1).abs()
        counts = ((distance < tolerance).sum(dim=1)) * usable
        top = int(counts.argmax())
        if int(counts[top]) > best_count:
            best_count, best = int(counts[top]), (origin[top], normal[top])
    if best is None or best_count <= 0:
        raise ValueError("No plane faces the cameras. Use the perspective projection for this scene.")

    point, normal = best
    for _ in range(3):
        inlier = ((points - point) @ normal).abs() < tolerance
        if int(inlier.sum()) < 3:
            break
        point = points[inlier].mean(dim=0)
        # The normal is the direction of least spread. A 3x3 eigenproblem avoids an SVD of a huge tall matrix.
        offset = (points[inlier] - point).double()
        normal = torch.linalg.eigh(offset.T @ offset)[1][:, 0].float()
    inlier = ((points - point) @ normal).abs() < tolerance
    return point, normal, inlier


def plan_plane(
    rec: Reconstruction,
    center: torch.Tensor,
    thresholds: torch.Tensor,
    far: float,
    scene_depth: float,
    settings: Settings,
) -> PlaneLayout:
    """Fit the dominant plane and look straight at it with an orthographic camera."""
    stride = max(1, rec.depth[0].numel() // 40000)
    clouds = [view_points(rec, i, center, far, float(thresholds[i]), False, stride) for i in range(len(rec.depth))]
    points, sizes = torch.cat([c.world for c in clouds]), torch.cat([c.size for c in clouds])
    facing = F.normalize(rec.extrinsic[:, 2, :3].mean(dim=0), dim=0)

    point, normal, inlier = fit_plane(points, 0.03 * scene_depth, facing)
    share = float(inlier.float().mean())
    if share < 0.15:
        raise ValueError(f"No dominant plane: only {100 * share:.0f}% of the points lie on one. Use the perspective projection.")
    if (camera_centers(rec.extrinsic).mean(dim=0) - point) @ normal < 0:
        normal = -normal
    forward = -normal

    up = F.normalize(-rec.extrinsic[:, 1, :3].mean(dim=0), dim=0)
    in_plane = up - (up @ normal) * normal
    if in_plane.norm() < 1e-3:
        in_plane = torch.linalg.cross(normal, rec.extrinsic[0, 0, :3])
    up = F.normalize(in_plane, dim=0)
    right = torch.linalg.cross(forward, up)

    offset = points - point
    a_min, a_max, y_min, y_max = _crop(offset @ right, offset @ up)
    stats = {
        "plane_inliers": f"{100 * share:.0f}%",
        "plane_tilt_deg": round(math.degrees(math.acos(min(1.0, abs(float(forward @ facing))))), 1),
    }
    ppr = settings.scale / float(sizes[inlier].median())
    layout = PlaneLayout(point, right, up, forward, a_min, a_max, y_min, y_max, ppr, stats)
    return _cap_resolution(layout, settings)


def plan_layout(
    rec: Reconstruction,
    center: torch.Tensor,
    thresholds: torch.Tensor,
    far: float,
    scene_depth: float,
    settings: Settings,
):
    if settings.projection == "plane":
        return plan_plane(rec, center, thresholds, far, scene_depth, settings)
    if settings.projection != "perspective":
        raise ValueError(f"Unknown projection {settings.projection!r}. Use one of {PROJECTIONS}.")
    return plan_perspective(rec, center, thresholds, far, settings)


# ---------------------------------------------------------------- rendering


def canvas_depth(
    rec: Reconstruction,
    layout,
    center: torch.Tensor,
    thresholds: torch.Tensor,
    far: float,
    use_sky: bool,
) -> torch.Tensor:
    """Depth of the surface closest to the virtual camera, per canvas pixel (H, W)."""
    height, width = layout.height, layout.width
    zbuf = torch.full((height * width,), math.inf)
    for i in range(len(rec.depth)):
        points = view_points(rec, i, center, far, float(thresholds[i]), use_sky)
        u, v, depth = layout.project(points.world)
        iu, iv = u.floor().long(), v.floor().long()
        radius = ((1.5 * layout.splat_pixels(points) - 1) / 2).ceil().clamp(0, 4).long()
        reach = int(radius.max()) if len(radius) else 0
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                x, y = iu + dx, iv + dy
                keep = (radius >= max(abs(dx), abs(dy))) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
                zbuf.scatter_reduce_(0, (y * width + x)[keep], depth[keep], reduce="amin")

    z = zbuf.view(1, 1, height, width)
    for _ in range(2):
        z = torch.where(torch.isinf(z), -F.max_pool2d(-z, 5, 1, 2), z)
    return z[0, 0]


class Blend(NamedTuple):
    color: torch.Tensor  # (3, H*W) weighted sum
    weight: torch.Tensor  # (H*W,)
    views: torch.Tensor  # (H*W,)
    view_weights: torch.Tensor | None  # (N, H*W)
    view_colors: torch.Tensor | None  # (N, 3, H*W)
    strict: torch.Tensor | None = None  # (H*W,) bool, a photo passed every test here


def fill_depth(z: torch.Tensor) -> torch.Tensor:
    """Fill empty z-buffer pixels (inf) with the average log-depth of nearby valid pixels."""
    known = torch.isfinite(z).numpy()
    if known.all() or not known.any():
        return z
    out = np.where(known, np.log(np.maximum(z.numpy(), 1e-6)), 0.0).astype(np.float32)
    known = known.copy()
    for radius in (4, 8, 16, 32, 64, 128, 256):
        size = (2 * radius + 1, 2 * radius + 1)
        numerator = cv2.blur(out * known, size, borderType=cv2.BORDER_CONSTANT)
        denominator = cv2.blur(known.astype(np.float32), size, borderType=cv2.BORDER_CONSTANT)
        newly = ~known & (denominator > 1e-3)
        out[newly] = numerator[newly] / denominator[newly]
        known |= newly
        if known.all():
            break
    return torch.where(torch.from_numpy(known), torch.from_numpy(np.exp(out)), torch.full_like(z, math.inf))


def blend(
    rec: Reconstruction,
    layout: Layout,
    zmap: torch.Tensor,
    center: torch.Tensor,
    thresholds: torch.Tensor,
    far: float,
    settings: Settings,
    gains: torch.Tensor,
    keep_views: bool = False,
    view_dtype: torch.dtype = torch.float32,
) -> Blend:
    """Look every canvas pixel up in every photo and average the photos that see the same surface."""
    count, _, height_src, width_src = rec.colors.shape
    height, width = zmap.shape
    total = height * width
    attributes = torch.stack(
        [rec.depth, rec.conf, torch.zeros_like(rec.depth) if rec.sky is None else rec.sky.float(), rec.edge.float()], dim=1
    )
    reference = torch.stack([torch.quantile(conf.reshape(-1)[::7], 0.9) for conf in rec.conf]).clamp_min(1.0 + 1e-3)
    ramp = max(settings.feather * min(height_src, width_src), 1.0)

    color_sum = torch.zeros(3, total)
    weight_sum = torch.zeros(total)
    views = torch.zeros(total)
    view_weights = torch.zeros(count, total) if keep_views else None
    view_colors = torch.zeros(count, 3, total, dtype=view_dtype) if keep_views else None
    # The fallback accumulates separately, so it never dilutes a pixel that has a strict match.
    fill = settings.fill_gaps
    zfill = fill_depth(zmap) if fill else zmap
    fb_color_sum = torch.zeros(3, total) if fill else None
    fb_weight_sum = torch.zeros(total) if fill else None
    fb_views = torch.zeros(total) if fill else None
    fb_view_weights = torch.zeros(count, total) if fill and keep_views else None

    rows = max(1, 1_500_000 // width)
    for top in range(0, height, rows):
        bottom = min(top + rows, height)
        v = (torch.arange(top, bottom) + 0.5)[:, None].expand(-1, width)
        u = (torch.arange(width) + 0.5)[None, :].expand(bottom - top, -1)
        z = zfill[top:bottom].reshape(-1)
        has_depth = torch.isfinite(zmap[top:bottom].reshape(-1))
        found = torch.isfinite(z)
        is_sky = found & (z >= 0.5 * far)
        point = layout.unproject(u.reshape(-1), v.reshape(-1), torch.where(found, z, torch.zeros_like(z)))
        span = slice(top * width, bottom * width)

        for i in range(count):
            rotation, translation = rec.extrinsic[i, :, :3], rec.extrinsic[i, :, 3]
            camera = point @ rotation.T + translation
            depth = camera[:, 2]
            safe = depth.clamp_min(1e-6)
            x = rec.intrinsic[i, 0, 0] * camera[:, 0] / safe + rec.intrinsic[i, 0, 2]
            y = rec.intrinsic[i, 1, 1] * camera[:, 1] / safe + rec.intrinsic[i, 1, 2]
            inside = found & (depth > 1e-6) & (x >= 0) & (x <= width_src) & (y >= 0) & (y <= height_src)
            grid = torch.stack([x / width_src * 2 - 1, y / height_src * 2 - 1], dim=-1).view(1, 1, -1, 2)
            src_depth, src_conf, src_sky, src_edge = F.grid_sample(
                attributes[i : i + 1], grid, mode="nearest", padding_mode="border", align_corners=False
            )[0, :, 0, :]
            color = F.grid_sample(rec.colors[i : i + 1], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0, :]

            solid = (src_depth - depth).abs() <= settings.depth_tolerance * safe
            solid = solid & (src_sky < 0.5) & (src_edge < 0.5) & (src_conf >= thresholds[i])
            visible = torch.where(is_sky, src_sky > 0.5, solid)
            confidence = torch.where(is_sky, torch.full_like(src_conf, 0.5), ((src_conf - 1) / (reference[i] - 1)).clamp(0, 1) + 0.05)
            border = (torch.minimum(x, width_src - x).minimum(torch.minimum(y, height_src - y)) / ramp).clamp(0, 1)
            smooth_border = border * border * (3 - 2 * border)
            weight = (inside & visible & has_depth).float() * confidence * smooth_border

            color_sum[:, span] += weight * color * gains[i][:, None]
            weight_sum[span] += weight
            views[span] += (weight > 1e-3).float()
            if keep_views:
                view_weights[i, span] = weight
                view_colors[i, :, span] = color
            if fill:
                # Any photo that sees the pixel, whatever its depth says. Photos nearer the middle count more.
                fallback = inside.float() * smooth_border
                fb_color_sum[:, span] += fallback * color * gains[i][:, None]
                fb_weight_sum[span] += fallback
                fb_views[span] += (fallback > 1e-3).float()
                if keep_views:
                    fb_view_weights[i, span] = fallback

    strict = weight_sum > 1e-3
    if fill:
        color_sum = torch.where(strict[None], color_sum, fb_color_sum)
        weight_sum = torch.where(strict, weight_sum, fb_weight_sum)
        views = torch.where(strict, views, fb_views)
        if keep_views:
            view_weights = torch.where(strict[None], view_weights, fb_view_weights)
    return Blend(color_sum, weight_sum, views, view_weights, view_colors, strict)


def estimate_gains(view_weights: torch.Tensor, view_colors: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """Per-photo, per-channel gains that make overlapping photos agree with their neighbours."""
    count = view_weights.shape[0]
    gains = torch.ones(count, 3)
    for _ in range(iterations):
        weighted = view_weights[:, None] * view_colors * gains[:, :, None]
        color_sum, weight_sum = weighted.sum(0), view_weights.sum(0)
        updated = gains.clone()
        for i in range(count):
            others = weight_sum - view_weights[i]
            overlap = (view_weights[i] > 1e-3) & (others > 1e-3)
            if int(overlap.sum()) < 200:
                continue
            reference = (color_sum - weighted[i])[:, overlap] / others[overlap]
            own = view_colors[i][:, overlap]
            weight = view_weights[i][overlap]
            updated[i] = (reference * weight).sum(1) / (own * weight).sum(1).clamp_min(1e-6)
        gains = (updated / updated.mean(0, keepdim=True)).clamp(0.5, 2.0)
    return gains


def multiband_blend(
    view_weights: torch.Tensor,
    view_colors: torch.Tensor,
    gains: torch.Tensor,
    fill: torch.Tensor,
    height: int,
    width: int,
    levels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Best photo per pixel, joined by Laplacian-pyramid blending (Burt and Adelson).

    Every pixel is assigned to the photo with the highest smoothed weight among the photos that
    pass the visibility test. A photo is blended with its neighbours over a wide band for the
    low frequencies and a narrow band for the details, so seams vanish without averaging two
    misaligned copies of the same detail into ghosts.

    `view_weights` is (N, H*W), `view_colors` (N, 3, H*W), `fill` (H, W, 3) is what a photo is
    painted with where it has no valid content, so black never bleeds in. Returns the image
    (H, W, 3), a bool mask of covered pixels (H, W), and the number of pyramid levels.
    """
    count = view_weights.shape[0]
    eligible = (view_weights > 1e-3).reshape(count, height, width)
    covered = eligible.any(dim=0)
    sigma = max(3.0, 0.01 * min(height, width))
    smooth = np.stack([cv2.GaussianBlur(w.reshape(height, width).numpy(), (0, 0), sigma) for w in view_weights])
    smooth = torch.from_numpy(smooth)
    smooth[~eligible] = -1.0
    label = smooth.argmax(dim=0)

    if levels is None:
        levels = int(min(6, max(1, math.log2(max(min(height, width), 2) / 24))))
    fill_np = fill.numpy().astype(np.float32)
    numerator = denominator = None
    for i in range(count):
        owned = (label == i) & covered
        if not owned.any():
            continue
        image = view_colors[i].float().reshape(3, height, width) * gains[i][:, None, None]
        image = torch.where(eligible[i][None], image, torch.from_numpy(fill_np).permute(2, 0, 1)).permute(1, 2, 0).contiguous().numpy()
        gaussian, masks = [image], [owned.float().numpy()]
        for _ in range(levels):
            gaussian.append(cv2.pyrDown(gaussian[-1]))
            masks.append(cv2.pyrDown(masks[-1]))
        laplacian = [
            gaussian[k] - cv2.pyrUp(gaussian[k + 1], dstsize=(gaussian[k].shape[1], gaussian[k].shape[0])) for k in range(levels)
        ] + [gaussian[-1]]
        if numerator is None:
            numerator = [np.zeros_like(band) for band in laplacian]
            denominator = [np.zeros(band.shape[:2], np.float32) for band in laplacian]
        for k in range(levels + 1):
            numerator[k] += laplacian[k] * masks[k][..., None]
            denominator[k] += masks[k]

    bands = [n / np.maximum(d, 1e-6)[..., None] for n, d in zip(numerator, denominator)]
    result = bands[-1]
    for k in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(bands[k].shape[1], bands[k].shape[0])) + bands[k]
    result = torch.from_numpy(np.clip(result, 0.0, 1.0))
    return torch.where(covered[..., None], result, fill), covered, levels


def render_panorama(rec: Reconstruction, settings: Settings) -> Panorama:
    started = time.time()
    if not settings.use_sky or settings.projection == "plane":
        rec = dataclasses.replace(rec, sky=None)
    count = len(rec.depth)
    centers = camera_centers(rec.extrinsic)
    center = choose_center(centers, settings.center)
    thresholds = confidence_thresholds(rec, settings.conf_percentile)

    solid = torch.isfinite(rec.depth) & (rec.depth > 0)
    if rec.sky is not None:
        solid = solid & ~rec.sky
    scene_depth = float(rec.depth[solid][::97].median()) if solid.any() else 1.0
    far = 1e3 * scene_depth

    layout = plan_layout(rec, center, thresholds, far, scene_depth, settings)
    gains = torch.ones(count, 3)
    if settings.exposure and count > 1:
        low = layout.with_ppr(layout.ppr * min(1.0, 1200 / layout.width))
        z_low = canvas_depth(rec, low, center, thresholds, far, settings.use_sky)
        probe = blend(rec, low, z_low, center, thresholds, far, settings, gains, keep_views=True)
        gains = estimate_gains(probe.view_weights, probe.view_colors)

    zmap = canvas_depth(rec, layout, center, thresholds, far, settings.use_sky)
    height, width = layout.height, layout.width
    use_bands = settings.blend == "multiband" and count > 1 and count * height * width <= _MULTIBAND_LIMIT
    result = blend(rec, layout, zmap, center, thresholds, far, settings, gains, keep_views=use_bands, view_dtype=torch.float16)

    rgb = (result.color / result.weight.clamp_min(1e-8)).T.reshape(height, width, 3).clamp(0, 1)
    blend_note = "average"
    if use_bands:
        rgb, _, levels = multiband_blend(result.view_weights, result.view_colors, gains, rgb, height, width)
        blend_note = f"multiband, {levels} levels"
    elif settings.blend == "multiband" and count > 1:
        blend_note = "average (canvas too large for multiband)"
    image = (rgb * 255).round().byte().numpy()
    alpha = ((result.weight > 1e-3).reshape(height, width).numpy() * 255).astype(np.uint8)
    closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    gaps = ((closed > 0) & (alpha == 0)).astype(np.uint8) * 255
    if gaps.any():
        image = cv2.inpaint(image, gaps, 3, cv2.INPAINT_TELEA)
        alpha = closed

    depth = zmap.clone()
    depth[(depth >= 0.5 * far) | torch.isinf(depth)] = math.nan
    spread = float(torch.cdist(centers, centers).max()) if count > 1 else 0.0
    info = {
        "photos": count,
        "canvas": f"{width} x {height}",
        "blend": blend_note,
        "filled_from_fallback": f"{100.0 * float((~result.strict & (result.weight > 1e-3)).float().mean()):.1f}%",
        "coverage": f"{100.0 * float((alpha > 0).mean()):.1f}%",
        "projection": settings.projection,
        **layout.describe(),
        "camera_spread_over_scene_depth": round(spread / scene_depth, 3),
        "exposure_gains": [[round(float(g), 3) for g in row] for row in gains] if settings.exposure else None,
        "render_seconds": round(time.time() - started, 1),
        "reconstruct_seconds": round(rec.timings.get("reconstruct", 0.0), 1),
    }
    return Panorama(image, alpha, result.views.reshape(height, width).byte().numpy(), depth.numpy(), info, layout)
