"""Synthetic checks for the panorama renderer. No model, no checkpoint.

The world is a sphere of radius 10 whose colour is a smooth function of the surface point.
Level cameras 30 degrees apart look around it. The renderer must reproduce that function on the
perspective canvas, with and without camera translation (parallax) and with garbage depth on sky
pixels.

    uv run python panorama/test_panorama.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stitch import Layout, PlaneLayout, Reconstruction, Settings, estimate_gains, fit_plane, multiband_blend, render_panorama

RADIUS = 10.0
HEIGHT, WIDTH, FOCAL = 96, 128, 80.0
STEP = 30  # degrees between neighbouring cameras
WALL_DEPTH = 3.0  # distance of the wall from the walking path


def texture(direction: torch.Tensor) -> torch.Tensor:
    """Smooth colour of a unit direction, shape (..., 3)."""
    x, y, z = direction.unbind(-1)
    channels = [0.5 + 0.45 * torch.sin(5 * x + 2 * c) * torch.cos(4 * y + 3 * z + c) for c in range(3)]
    return torch.stack(channels, dim=-1)


def make_scene(num_cameras: int = 3, orbit: float = 0.0, sky_rows: int = 0, seed: int = 0) -> Reconstruction:
    generator = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(torch.arange(HEIGHT) + 0.5, torch.arange(WIDTH) + 0.5, indexing="ij")
    ray = torch.stack([(xs - WIDTH / 2) / FOCAL, (ys - HEIGHT / 2) / FOCAL, torch.ones_like(xs)], dim=-1)
    unit = F.normalize(ray, dim=-1)

    colors, depths, extrinsics = [], [], []
    for k in range(num_cameras):
        yaw = math.radians(STEP * k)
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        center = orbit * torch.tensor([math.sin(yaw), 0.0, math.cos(yaw)])
        world = unit @ rotation  # R^T d for row vectors
        along = -(world @ center) + torch.sqrt((world @ center) ** 2 - center @ center + RADIUS**2)
        point = center + world * along[..., None]
        color = texture(F.normalize(point, dim=-1))
        depth = along / ray.norm(dim=-1)
        if sky_rows:
            color[:sky_rows] = texture(world[:sky_rows])
            depth[:sky_rows] = 0.05 + torch.rand(sky_rows, WIDTH, generator=generator)
        colors.append(color.permute(2, 0, 1))
        depths.append(depth)
        extrinsics.append(torch.cat([rotation, (-rotation @ center)[:, None]], dim=1))

    intrinsic = torch.tensor([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1.0]])
    depth = torch.stack(depths)
    sky = None
    if sky_rows:
        sky = torch.zeros(num_cameras, HEIGHT, WIDTH, dtype=torch.bool)
        sky[:, :sky_rows] = True
    return Reconstruction(
        colors=torch.stack(colors),
        depth=depth,
        conf=torch.full_like(depth, 5.0),
        edge=torch.zeros_like(depth, dtype=torch.bool),
        sky=sky,
        extrinsic=torch.stack(extrinsics),
        intrinsic=intrinsic.expand(num_cameras, -1, -1).contiguous(),
    )


def expected_image(panorama, center: torch.Tensor | None = None) -> torch.Tensor:
    """Texture of the sphere point each canvas ray hits, from the virtual camera at `center`."""
    height, width = panorama.image.shape[:2]
    v, u = torch.meshgrid(torch.arange(height) + 0.5, torch.arange(width) + 0.5, indexing="ij")
    direction = panorama.layout.to_directions(u, v)
    if center is None:
        return texture(direction)
    along = -(direction @ center) + torch.sqrt((direction @ center) ** 2 - center @ center + RADIUS**2)
    return texture(F.normalize(center + direction * along[..., None], dim=-1))


def error_inside(panorama, center: torch.Tensor | None = None) -> float:
    """Mean absolute colour error over covered pixels, away from the canvas border."""
    height, width = panorama.image.shape[:2]
    covered = torch.from_numpy(panorama.alpha > 0)
    interior = torch.zeros_like(covered)
    interior[height // 20 : -height // 20, width // 40 : -width // 40] = True
    mask = covered & interior
    got = torch.from_numpy(panorama.image).float() / 255
    return float((got - expected_image(panorama, center)).abs()[mask].mean())


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"ok  {name}")


def test_rotation_only_panorama() -> None:
    panorama = render_panorama(make_scene(3), Settings(exposure=False))
    error = error_inside(panorama)
    fov = panorama.info["horizontal_fov_deg"]
    check(f"error {error:.4f}", error < 0.02)
    check("photos overlap", int(panorama.views.max()) >= 2)
    check(f"canvas spans the photos ({fov} deg)", 130 < fov < 145)


def test_canvas_is_centred_on_the_photos() -> None:
    panorama = render_panorama(make_scene(3), Settings(exposure=False))
    layout = panorama.layout
    check("canvas is symmetric around the axis", abs(layout.a_min + layout.a_max) < 0.1)
    one = render_panorama(make_scene(1), Settings(exposure=False))
    check(f"one photo keeps its own field of view ({one.info['horizontal_fov_deg']} deg)", 74 < one.info["horizontal_fov_deg"] < 82)


def test_parallax_is_handled() -> None:
    scene = make_scene(3, orbit=0.8)
    centers = -torch.einsum("nji,nj->ni", scene.extrinsic[:, :, :3], scene.extrinsic[:, :, 3])
    panorama = render_panorama(scene, Settings(exposure=False))
    error = error_inside(panorama, centers.mean(dim=0))
    check(f"translated cameras: error {error:.4f}", error < 0.03)
    check("coverage is high", float((panorama.alpha > 0).mean()) > 0.7)


def test_sky_with_garbage_depth() -> None:
    error = error_inside(render_panorama(make_scene(3, sky_rows=30), Settings(exposure=False, use_sky=True)))
    check(f"sky handled: error {error:.4f}", error < 0.03)


def test_exposure_gains_recover_the_scale() -> None:
    scene = make_scene(3)
    factors = torch.tensor([0.7, 1.0, 1.3])
    scene.colors = (scene.colors * factors[:, None, None, None]).clamp(0, 1)
    plain = render_panorama(scene, Settings(exposure=False))
    corrected = render_panorama(scene, Settings(exposure=True))
    check("gains reduce the seam variance", error_inside(corrected) < error_inside(plain))

    weights = torch.ones(3, 1000)
    base = torch.rand(3, 1000) * 0.5 + 0.2
    colors = torch.stack([base, base * 1.2, base * 0.8])
    gains = estimate_gains(weights, colors)
    check("gains are relative to the mean", abs(float(gains.mean()) - 1.0) < 0.05 and float(gains[1, 0]) < float(gains[0, 0]))


def straightness(u: torch.Tensor, v: torch.Tensor) -> float:
    """Largest distance of the points from their best-fit line, in pixels."""
    points = torch.stack([u, v], dim=1)
    points = points - points.mean(dim=0)
    direction = torch.linalg.svd(points)[2][0]
    normal = torch.stack([-direction[1], direction[0]])
    return float((points @ normal).abs().max())


def test_straight_lines_stay_straight() -> None:
    layout = Layout(torch.eye(3), -2.0, 2.0, -2.0, 2.0, 200.0)
    t = torch.linspace(-0.5, 0.5, 200)
    for name, line in (
        ("slanted", torch.stack([0.3 + t, -0.2 + 0.4 * t, 1.0 + 0.1 * t], dim=1)),
        ("vertical", torch.stack([torch.full_like(t, 0.5), t, torch.ones_like(t)], dim=1)),
        ("horizontal", torch.stack([t, torch.full_like(t, 0.3), torch.ones_like(t) * 1.2], dim=1)),
    ):
        u, v = layout.to_canvas(F.normalize(line, dim=1))
        check(f"{name} line ({straightness(u, v):.3f} px)", straightness(u, v) < 0.05)

    generator = torch.Generator().manual_seed(3)
    lon = (torch.rand(500, generator=generator) - 0.5) * math.radians(120)
    lat = (torch.rand(500, generator=generator) - 0.5) * math.radians(80)
    direction = torch.stack([torch.cos(lat) * torch.sin(lon), -torch.sin(lat), torch.cos(lat) * torch.cos(lon)], dim=1)
    back = layout.to_directions(*layout.to_canvas(direction))
    check("canvas round trip", float((back - direction).abs().max()) < 1e-4)


def test_multiband_joins_without_a_step() -> None:
    height, width = 120, 200
    weights = torch.zeros(2, height * width)
    left = torch.arange(width)[None, :].expand(height, -1) < width // 2
    weights[0] = left.float().reshape(-1) * 0.9 + 0.1
    weights[1] = (~left).float().reshape(-1) * 0.9 + 0.1
    colors = torch.stack([torch.full((3, height * width), 0.3), torch.full((3, height * width), 0.7)])
    image, covered, levels = multiband_blend(weights, colors, torch.ones(2, 3), torch.zeros(height, width, 3), height, width)
    row = image[height // 2, :, 0]
    check(f"left and right keep their colour ({float(row[5]):.2f}, {float(row[-5]):.2f})", abs(float(row[5]) - 0.3) < 0.03 and abs(float(row[-5]) - 0.7) < 0.03)
    jump = float((row[1:] - row[:-1]).abs().max())
    check(f"the seam is a ramp, not a step (largest jump {jump:.3f})", jump < 0.05)
    check("everything is covered", bool(covered.all()) and levels >= 1)

    black = torch.zeros(2, 3, height * width)
    image, _, _ = multiband_blend(weights, black, torch.ones(2, 3), torch.full((height, width, 3), 0.5), height, width)
    check("output stays in range", float(image.min()) >= 0.0 and float(image.max()) <= 1.0)


def test_multiband_render_matches_the_scene() -> None:
    scene = make_scene(3)
    for blend in ("multiband", "average"):
        panorama = render_panorama(scene, Settings(exposure=False, blend=blend))
        error = error_inside(panorama)
        check(f"{blend}: error {error:.4f}", error < 0.02)
    check("multiband is reported", "multiband" in render_panorama(scene, Settings(exposure=False)).info["blend"])
    factors = torch.tensor([0.8, 1.0, 1.2])
    scene.colors = (scene.colors * factors[:, None, None, None]).clamp(0, 1)
    mismatched = render_panorama(scene, Settings(exposure=True, blend="multiband"))
    check(f"multiband with exposure matching: error {error_inside(mismatched):.4f}", error_inside(mismatched) < 0.03)


def test_gaps_are_filled_from_the_best_photo() -> None:
    scene = make_scene(1)
    scene.conf[0, 30:60, 40:90] = 1.0  # a block the model does not trust, so the strict tests reject it
    strict = render_panorama(scene, Settings(exposure=False, fill_gaps=False))
    filled = render_panorama(scene, Settings(exposure=False, fill_gaps=True))
    holes = float((strict.alpha == 0).mean())
    check(f"wrong depth leaves holes without the fill ({100 * holes:.1f}%)", holes > float((filled.alpha == 0).mean()) + 0.005)
    check(f"the fill closes them ({filled.info['filled_from_fallback']} from fallback)", float((filled.alpha == 0).mean()) < holes)
    check(f"filled panorama stays accurate: error {error_inside(filled):.4f}", error_inside(filled) < 0.03)
    clean = render_panorama(make_scene(3), Settings(exposure=False, fill_gaps=True))
    check("a clean scene needs no fallback", clean.info["filled_from_fallback"] == "0.0%")


def test_depth_holes_are_filled() -> None:
    from stitch import fill_depth

    z = torch.full((40, 60), 5.0)
    z[10:20, 10:30] = math.inf
    filled = fill_depth(z)
    check("holes get the surrounding depth", bool(torch.isfinite(filled).all()) and abs(float(filled[15, 20]) - 5.0) < 0.01)
    check("nothing to fill returns the input", torch.equal(fill_depth(torch.ones(4, 4)), torch.ones(4, 4)))


def plane_texture(points: torch.Tensor, tilt: float) -> torch.Tensor:
    """Colour of a point of the wall: a smooth function of its two coordinates in the wall."""
    along = torch.tensor([math.cos(tilt), 0.0, math.sin(tilt)])
    offset = points - torch.tensor([0.0, 0.0, WALL_DEPTH])
    a, y = offset @ along, offset[..., 1]
    return texture(torch.stack([0.4 * a, 0.4 * y, torch.full_like(a, 0.3)], dim=-1))


def make_wall(num_cameras: int = 5, spacing: float = 1.5, tilt_deg: float = 0.0, wobble: float = 0.0) -> Reconstruction:
    """A textured wall seen by level cameras that walk along it (translation, no rotation)."""
    tilt = math.radians(tilt_deg)
    normal = torch.tensor([math.sin(tilt), 0.0, -math.cos(tilt)])  # towards the cameras
    wall_point = torch.tensor([0.0, 0.0, WALL_DEPTH])
    ys, xs = torch.meshgrid(torch.arange(HEIGHT) + 0.5, torch.arange(WIDTH) + 0.5, indexing="ij")
    ray = torch.stack([(xs - WIDTH / 2) / FOCAL, (ys - HEIGHT / 2) / FOCAL, torch.ones_like(xs)], dim=-1)

    colors, depths, extrinsics = [], [], []
    for k in range(num_cameras):
        center = torch.tensor([spacing * k, wobble * math.sin(k * 1.3), 0.0])
        along = ((wall_point - center) @ normal) / (ray @ normal)  # camera depth of the wall point (ray z is 1)
        point = center + ray * along[..., None]
        colors.append(plane_texture(point, tilt).permute(2, 0, 1))
        depths.append(along)
        extrinsics.append(torch.cat([torch.eye(3), -center[:, None]], dim=1))

    intrinsic = torch.tensor([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1.0]])
    depth = torch.stack(depths)
    return Reconstruction(
        colors=torch.stack(colors),
        depth=depth,
        conf=torch.full_like(depth, 5.0),
        edge=torch.zeros_like(depth, dtype=torch.bool),
        sky=None,
        extrinsic=torch.stack(extrinsics),
        intrinsic=intrinsic.expand(num_cameras, -1, -1).contiguous(),
    )


def wall_error(panorama, tilt_deg: float = 0.0) -> float:
    height, width = panorama.image.shape[:2]
    v, u = torch.meshgrid(torch.arange(height) + 0.5, torch.arange(width) + 0.5, indexing="ij")
    wall = panorama.layout.unproject(u, v, torch.zeros_like(u))
    covered = torch.from_numpy(panorama.alpha > 0)
    interior = torch.zeros_like(covered)
    interior[height // 20 : -height // 20, width // 40 : -width // 40] = True
    mask = covered & interior
    got = torch.from_numpy(panorama.image).float() / 255
    return float((got - plane_texture(wall, math.radians(tilt_deg))).abs()[mask].mean())


def test_plane_mosaic_of_a_walk_along_a_wall() -> None:
    panorama = render_panorama(make_wall(), Settings(projection="plane", exposure=False))
    error = wall_error(panorama)
    check(f"error {error:.4f}", error < 0.02)
    check("photos overlap", int(panorama.views.max()) >= 2)
    layout = panorama.layout
    check(f"canvas is as wide as the walk ({layout.a_max - layout.a_min:.2f} units)", abs((layout.a_max - layout.a_min) - 10.8) < 0.5)
    check("the wall is rendered upright (right is +x)", float(layout.right[0]) > 0.99 and float(layout.up[1]) < -0.99)
    check(f"almost no empty pixels ({float((panorama.alpha == 0).mean()):.3f})", float((panorama.alpha == 0).mean()) < 0.05)


def test_plane_has_the_same_scale_everywhere() -> None:
    # A wall that is not square to the walking direction: the case a pinhole canvas foreshortens.
    tilt = math.radians(25.0)
    wall = make_wall(tilt_deg=25.0)
    plane = render_panorama(wall, Settings(projection="plane", exposure=False)).layout
    perspective = render_panorama(wall, Settings(projection="perspective", exposure=False)).layout
    along = torch.tensor([math.cos(tilt), 0.0, math.sin(tilt)])

    def spacing(layout, s: float) -> float:
        start = torch.tensor([0.0, 0.0, WALL_DEPTH]) + s * along
        u, v, _ = layout.project(torch.stack([start, start + 0.1 * along]))
        return float(torch.hypot(u[1] - u[0], v[1] - v[0]))

    plane_ratio = spacing(plane, 4.0) / spacing(plane, -3.0)
    perspective_ratio = spacing(perspective, 4.0) / spacing(perspective, -3.0)
    check(f"plane: 10 cm of wall is the same size at both ends (ratio {plane_ratio:.3f})", abs(plane_ratio - 1.0) < 0.01)
    check(f"perspective foreshortens the far end (ratio {perspective_ratio:.2f})", abs(math.log(perspective_ratio)) > 0.5)


def test_plane_follows_a_tilted_wall() -> None:
    panorama = render_panorama(make_wall(tilt_deg=20.0), Settings(projection="plane", exposure=False))
    error = wall_error(panorama, 20.0)
    tilt = panorama.info["plane_tilt_deg"]
    check(f"error {error:.4f}", error < 0.03)
    check(f"the wall is found {tilt} deg off the view direction", 17 < tilt < 23)


def test_plane_survives_cameras_that_bob() -> None:
    panorama = render_panorama(make_wall(wobble=0.15), Settings(projection="plane", exposure=False))
    error = wall_error(panorama)
    check(f"error {error:.4f}", error < 0.03)


def test_plane_fit_ignores_objects_in_front() -> None:
    generator = torch.Generator().manual_seed(1)
    tilt = math.radians(20.0)
    along = torch.tensor([math.cos(tilt), 0.0, math.sin(tilt)])
    on_wall = (torch.rand(3000, 1, generator=generator) - 0.5) * 8 * along + torch.tensor([0.0, 0.0, WALL_DEPTH])
    on_wall[:, 1] = (torch.rand(3000, generator=generator) - 0.5) * 4
    clutter = torch.rand(1300, 3, generator=generator) * torch.tensor([8.0, 4.0, 2.0]) + torch.tensor([-4.0, -2.0, 1.0])
    point, normal, inlier = fit_plane(torch.cat([on_wall, clutter]), 0.05, torch.tensor([0.0, 0.0, 1.0]))
    expected = torch.tensor([math.sin(tilt), 0.0, -math.cos(tilt)])
    angle = math.degrees(math.acos(min(1.0, abs(float(normal @ expected)))))
    check(f"normal is within {angle:.2f} deg of the wall", angle < 2.0)
    check("most wall points are inliers", float(inlier[:3000].float().mean()) > 0.9)


def test_plane_refuses_a_scene_without_one() -> None:
    scene = make_scene(3)
    scene.depth = torch.rand(scene.depth.shape, generator=torch.Generator().manual_seed(2)) * 10 + 1
    try:
        render_panorama(scene, Settings(projection="plane", exposure=False))
    except ValueError as error:
        check(f"clear message: {str(error)[:60]}", "plane" in str(error).lower())
    else:
        check("a scene without a plane is refused", False)


def main() -> None:
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            print(f"-- {name}")
            test()
    print("panorama checks passed")


if __name__ == "__main__":
    main()
