"""CPU checks for the multi-scene pieces. No checkpoint and no dataset required.

    uv run python training/multiscene_smoke.py
    uv run pytest training/multiscene_smoke.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from check_single_scene import prediction_gap
from data.mixed_scene import MixedSceneCollate
from losses.base_loss.affinity import balanced_bce, compute_affinity_loss
from losses.loss import MultitaskLoss
from losses.per_scene import anchor_to_first_frame, mean_over_scenes
from multiscene_eval import clustering_scores, depth_scores, pose_scores, run_systems
from vggt_omega.models import VGGTOmega
from vggt_omega.models.aggregator import Aggregator, _apply_by_group
from vggt_omega.models.multiscene import (
    AffinityHead,
    MultiScene,
    SceneAttention,
    choose_groups,
    predict_pose_by_scene,
    scene_groups,
)
from vggt_omega.models.vggt_omega import _build_multiscene
from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding
from vggt_omega.utils.scenes import (
    adjusted_rand_index,
    anchor_offset,
    pack_scenes,
    points_in_anchor_frame,
    poses_relative_to_anchor,
)

TINY = dict(embed_dim=64, autocast=False)
TINY_SCENE = {"enabled": True, "num_heads": 4, "route_layer": 3, "gt_group_prob": [0.0, 0.0]}


def _check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"ok  {name}")


def _tiny_model(**scene_overrides) -> VGGTOmega:
    torch.manual_seed(0)
    return VGGTOmega(**TINY, multiscene={**TINY_SCENE, **scene_overrides}).eval()


def _tiny_bag(num_frames: int = 6, size: int = 32) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(1, num_frames, 3, size, size)


class _MeanCamera(torch.nn.Module):
    """Stand-in camera head: every frame receives the mean of the frames it was shown."""

    def forward(self, aggregated_tokens_list, patch_token_start):
        tokens = aggregated_tokens_list[-1]
        mixed = tokens[:, :, 0, :9].mean(dim=1, keepdim=True)
        return mixed.expand(tokens.shape[0], tokens.shape[1], 9).contiguous()


# ---------------------------------------------------------------- affinity and grouping


def test_affinity_head_can_compare_frames() -> None:
    torch.manual_seed(0)
    head = AffinityHead(dim=32)
    centers = torch.randn(3, 32) * 2
    scene_id = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    target = (scene_id[:, None] == scene_id[None, :]).float()[None]
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(300):
        tokens = (centers[scene_id] + 0.3 * torch.randn(12, 32))[None, :, None, :].expand(1, 12, 16, 32)
        _, logits = head(tokens)
        loss = balanced_bce(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    affinity, _ = head(tokens)
    groups = scene_groups(affinity.detach(), 0.5)[0]
    _check("three scenes recovered", adjusted_rand_index(groups, scene_id) > 0.99)


def test_untrained_heads_call_everything_one_scene() -> None:
    block = MultiScene(dim=32, num_heads=4)
    tokens = torch.randn(2, 6, 18, 32)
    route = block.route(tokens[:, :, 1:17])
    _check("route groups", torch.equal(route["route_group_id"], torch.zeros(2, 6, dtype=torch.long)))
    _, out = block(tokens, patch_token_start=17)
    _check("final groups", torch.equal(out["group_id"], torch.zeros(2, 6, dtype=torch.long)))
    affinity = out["affinity"]
    _check("symmetric", torch.allclose(affinity, affinity.transpose(-1, -2), atol=1e-6))
    _check("unit diagonal", torch.allclose(affinity.diagonal(dim1=-2, dim2=-1), torch.ones(2, 6)))


def test_grouping() -> None:
    two = torch.tensor([[[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]]])
    _check("two groups", torch.equal(scene_groups(two, 0.5), torch.tensor([[0, 0, 1]])))

    # One false-positive pair (frames 2 and 3) between two clear scenes.
    noisy = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )
    noisy[1, 2] = noisy[2, 1] = 0.6
    average = scene_groups(noisy[None], 0.5, "average")
    single = scene_groups(noisy[None], 0.5, "single")
    _check("average linkage keeps the scenes apart", torch.equal(average, torch.tensor([[0, 0, 1, 1]])))
    _check("single linkage merges them", int(single.max()) == 0)

    labels = scene_groups(torch.eye(4)[None], 0.5)
    _check("all singletons", torch.equal(labels, torch.arange(4)[None]))
    _check("ids follow first frame", torch.equal(scene_groups(torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]), 0.5), torch.tensor([[0, 1, 0]])))


def test_adjusted_rand_index() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    _check("identical", adjusted_rand_index(labels, labels) == 1.0)
    _check("relabeled", adjusted_rand_index(labels, torch.tensor([5, 5, 3, 3, 9, 9])) == 1.0)
    _check("all merged is poor", adjusted_rand_index(torch.zeros(6, dtype=torch.long), labels) < 0.1)
    _check("single frame", adjusted_rand_index(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)) == 1.0)


def test_scene_attention_respects_groups() -> None:
    torch.manual_seed(0)
    block = SceneAttention(dim=32, num_heads=4)
    for layer in (block.proj, block.mlp[-1]):
        torch.nn.init.normal_(layer.weight, std=0.1)
    tokens = torch.randn(1, 4, 16, 32)
    same = torch.tensor([[0, 0, 1, 1]])
    mask = same[:, :, None] == same[:, None, :]
    base = block(tokens, mask)
    changed = tokens.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:])
    after = block(changed, mask)
    _check("scene 0 ignores scene 1", torch.allclose(base[:, :2], after[:, :2], atol=1e-5))
    _check("scene 1 changed", not torch.allclose(base[:, 2:], after[:, 2:], atol=1e-3))

    fresh = SceneAttention(dim=32, num_heads=4)
    _check("zero residual at init", torch.allclose(fresh(tokens, mask), tokens, atol=1e-6))


def test_choose_groups() -> None:
    predicted = torch.tensor([[0, 0, 1, 1]])
    labels = torch.tensor([[1, 1, 0, 0]])
    _check("gt when asked", torch.equal(choose_groups(predicted, labels, True), labels))
    _check("predicted otherwise", torch.equal(choose_groups(predicted, labels, False), predicted))
    _check("no labels at inference", torch.equal(choose_groups(predicted, None, True), predicted))


# ---------------------------------------------------------------- routing in the backbone


def test_apply_by_group() -> None:
    tokens = torch.arange(4.0).view(1, 4, 1, 1)

    def add_group_mean(part):
        return part + part.mean(dim=1, keepdim=True)

    result = _apply_by_group(tokens, torch.tensor([[0, 1, 0, 1]]), add_group_mean)
    _check("per-group mean", torch.allclose(result.flatten(), torch.tensor([0 + 1.0, 1 + 2.0, 2 + 1.0, 3 + 2.0])))
    one = _apply_by_group(tokens, torch.zeros(1, 4, dtype=torch.long), add_group_mean)
    _check("one group is the plain call", torch.allclose(one, add_group_mean(tokens)))


def test_aggregator_attends_inside_groups_after_the_route_layer() -> None:
    torch.manual_seed(0)
    aggregator = Aggregator(embed_dim=64).eval()
    seen = []
    original = aggregator._inter_frame_attention

    def spy(tokens, block_idx, attention_type):
        seen.append((block_idx, tokens.shape[1]))
        return original(tokens, block_idx, attention_type)

    aggregator._inter_frame_attention = spy
    groups = torch.tensor([[0, 0, 1, 1, 1, 2]])
    with torch.no_grad():
        aggregator(_tiny_bag(), router=lambda scene_tokens: groups, route_layer=5)
    before = {frames for index, frames in seen if index <= 5}
    after = {frames for index, frames in seen if index > 5}
    _check("full bag up to the route layer", before == {6})
    _check("groups after it", after == {2, 3, 1})


# ---------------------------------------------------------------- model


def test_single_scene_matches_the_original_model_at_init() -> None:
    reference = VGGTOmega(**TINY).eval()
    candidate = VGGTOmega(**TINY, multiscene=TINY_SCENE).eval()
    missing, unexpected = candidate.load_state_dict(reference.state_dict(), strict=False)
    _check("only new keys are missing", not unexpected and all(key.startswith("multiscene.") for key in missing))
    images = _tiny_bag()
    with torch.no_grad():
        expected = reference(images)
        actual = candidate(images)
    gap = prediction_gap(expected, actual)
    _check("pose matches", gap["pose_enc"]["max"] < 1e-4)
    _check("depth matches", gap["depth"]["max"] < 1e-4)


def test_model_outputs_and_modes() -> None:
    model = _tiny_model()
    images = _tiny_bag()
    scene_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    with torch.no_grad():
        predicted = model(images)
        oracle = model(images, scene_id=scene_id, use_gt_groups=True)
        plain = model(images, isolate=False)
    for key in ("affinity", "route_affinity", "group_id", "route_group_id", "used_group_id", "pose_enc", "depth"):
        _check(f"output {key}", key in predicted)
    _check("oracle uses labels", torch.equal(oracle["used_group_id"], scene_id) and float(oracle["used_gt_groups"]) == 1.0)
    _check("plain is one group", int(plain["used_group_id"].max()) == 0)
    _check("scene ids ignored without the flag", float(model(images, scene_id=scene_id)["used_gt_groups"]) == 0.0)


def test_group_probability_schedule() -> None:
    model = _tiny_model(gt_group_prob=[1.0, 0.0])
    _check("starts at 1", model.gt_group_prob == 1.0)
    model.set_group_progress(0.25)
    _check("linear", abs(model.gt_group_prob - 0.75) < 1e-9)
    model.set_group_progress(2.0)
    _check("clamped", model.gt_group_prob == 0.0)
    model.train()
    model.set_group_progress(0.0)
    images, scene_id = _tiny_bag(), torch.tensor([[0, 0, 0, 1, 1, 1]])
    with torch.no_grad():
        _check("training draws labels", float(model(images, scene_id=scene_id)["used_gt_groups"]) == 1.0)


def test_predict_scenes() -> None:
    model = _tiny_model()
    scenes = model.predict_scenes(_tiny_bag()[0])
    _check("one scene at init", len(scenes) == 1 and scenes[0]["images"] == list(range(6)))
    first = scenes[0]
    _check("pose shape", first["camera_poses"].shape == (6, 3, 4))
    _check("anchor pose is identity", torch.allclose(first["camera_poses"][first["images"].index(first["anchor"])], torch.eye(3, 4), atol=1e-4))
    _check("pointcloud shape", tuple(first["pointcloud"].shape) == (6, 32, 32, 3))


def test_camera_head_stays_inside_its_scene() -> None:
    tokens = torch.zeros(1, 4, 2, 9)
    tokens[0, :2, 0, 0] = 1
    tokens[0, 2:, 0, 0] = 10
    groups = torch.tensor([[0, 0, 1, 1]])
    pose = predict_pose_by_scene(_MeanCamera(), tokens, patch_token_start=1, group_id=groups)
    _check("scene 0 pose", torch.allclose(pose[0, :2, 0], torch.ones(2)))
    _check("scene 1 pose", torch.allclose(pose[0, 2:, 0], torch.full((2,), 10.0)))

    tokens = torch.randn(1, 4, 2, 9, requires_grad=True)
    pose = predict_pose_by_scene(_MeanCamera(), tokens, patch_token_start=1, group_id=groups)
    pose[0, 0].sum().backward()
    _check("grad stays in scene 0", tokens.grad[0, 2:].abs().sum().item() == 0)

    whole = torch.zeros(1, 4, dtype=torch.long)
    together = predict_pose_by_scene(_MeanCamera(), tokens.detach(), patch_token_start=1, group_id=whole)
    _check("one group matches the plain head", torch.allclose(together, _MeanCamera()([tokens.detach()], patch_token_start=1)))


def test_disabled_block_is_absent() -> None:
    _check("unset", _build_multiscene(None, dim=32) is None)
    _check("disabled", _build_multiscene({"enabled": False}, dim=32) is None)
    built = _build_multiscene({"enabled": True, "num_heads": 4, "threshold": 0.4, "route_layer": 7}, dim=32)
    _check("enabled", isinstance(built, MultiScene) and built.threshold == 0.4 and built.route_layer == 7)


# ---------------------------------------------------------------- data


def _toy_scene(name: str, dataset: str = "toy", frames: int = 5) -> dict:
    return {
        "images": torch.rand(frames, 3, 8, 8),
        "depths": torch.rand(frames, 8, 8),
        "extrinsics": torch.eye(3, 4).expand(frames, -1, -1).contiguous(),
        "intrinsics": torch.eye(3).expand(frames, -1, -1).contiguous(),
        "world_points": torch.rand(frames, 8, 8, 3),
        "point_masks": torch.ones(frames, 8, 8, dtype=torch.bool),
        "seq_name": name,
        "dataset": dataset,
        "is_synthetic": False,
        "depth_train_mask": True,
    }


def test_collate() -> None:
    samples = [_toy_scene(name) for name in "abcd"]
    batch = MixedSceneCollate(num_scenes=2, shuffle_frames=False)(samples)
    _check("one bag", batch["images"].shape == (1, 10, 3, 8, 8))
    _check("scene ids", torch.equal(batch["scene_id"][0], torch.tensor([0] * 5 + [1] * 5)))
    _check("affinity", batch["affinity"].shape == (1, 10, 10) and float(batch["affinity"][0, 0, 9]) == 0.0)

    shuffled = MixedSceneCollate(num_scenes=2, shuffle_frames=True)(samples)
    _check("shuffle keeps frames and labels aligned", shuffled["images"].shape[1] == 10 and shuffled["scene_id"].sum() == 5)

    torch.manual_seed(0)
    labels = torch.stack([MixedSceneCollate(num_scenes=2, shuffle_frames=True)(samples)["scene_id"][0] for _ in range(8)])
    _check("order is random", len({tuple(row.tolist()) for row in labels}) > 1)

    ranged = MixedSceneCollate(num_scenes_range=[1, 3], frames_range=[3, 4], max_total_frames=8)
    counts, sizes = set(), set()
    for _ in range(60):
        bag = ranged(samples)
        counts.add(int(bag["num_scenes"][0]))
        sizes.add(int(bag["num_frames"][0]))
        _check("cap respected", int(bag["num_frames"][0]) <= 8)
    _check("scene count varies", 1 in counts and len(counts) > 1)
    _check("frame count varies", len(sizes) > 1)

    duplicates = [_toy_scene("a"), _toy_scene("a"), _toy_scene("b")]
    _check("duplicate sequence dropped", int(MixedSceneCollate(num_scenes=2)(duplicates)["num_scenes"][0]) == 2)
    try:
        MixedSceneCollate(num_scenes=2)([_toy_scene("a"), _toy_scene("a")])
    except ValueError:
        _check("too few distinct sequences", True)
    else:
        _check("too few distinct sequences", False)

    mixed = [_toy_scene("a", "x"), _toy_scene("b", "y"), _toy_scene("c", "x"), _toy_scene("d", "y")]
    same = MixedSceneCollate(num_scenes=2, same_dataset_prob=1.0)
    names = {same(mixed)["dataset"][0] for _ in range(20)}
    _check("same-dataset bags", names <= {"x|x", "y|y"})


# ---------------------------------------------------------------- losses


def _bag_batch(num_frames: int = 6, size: int = 32) -> dict:
    torch.manual_seed(2)
    scene_id = torch.tensor([[0, 0, 0, 1, 1, 1]])[:, :num_frames]
    extrinsics = torch.eye(3, 4).repeat(1, num_frames, 1, 1)
    extrinsics[0, :, 0, 3] = torch.linspace(0, 1, num_frames)
    return {
        "scene_id": scene_id,
        "affinity": (scene_id[:, :, None] == scene_id[:, None, :]).float(),
        "images": torch.rand(1, num_frames, 3, size, size),
        "extrinsics": extrinsics,
        "intrinsics": torch.tensor([[size, 0, size / 2], [0, size, size / 2], [0, 0, 1.0]]).repeat(1, num_frames, 1, 1),
        "valid_seq_mask": torch.ones(1, dtype=torch.bool),
        "depths": torch.rand(1, num_frames, size, size) + 0.5,
        "point_masks": torch.ones(1, num_frames, size, size, dtype=torch.bool),
        "depth_train_mask": torch.ones(1, dtype=torch.bool),
        "is_synthetic": torch.zeros(1, dtype=torch.bool),
    }


def _criterion(**kwargs) -> MultitaskLoss:
    camera = {"weight": 1.0, "weight_pairwise": 1.0, "normalize_trans_by_gt_scale": False}
    depth = {"weight": 1.0, "min_valid_pts": 1, "use_conf_loss": False, "gradient_loss_config": None, "weight_normal": 0.0}
    point = {"weight": 1.0, "min_valid_pts": 1}
    return MultitaskLoss(camera=camera, depth=depth, point=point, **kwargs)


def test_balanced_bce_and_affinity_loss() -> None:
    target = torch.eye(4).unsqueeze(0)
    _check("one scene has no negatives", torch.isfinite(balanced_bce(torch.zeros(1, 4, 4), torch.ones(1, 4, 4))))
    _check("single frame", float(balanced_bce(torch.zeros(1, 1, 1), torch.ones(1, 1, 1))) == 0.0)

    class_imbalanced = torch.zeros(1, 12, 12)
    labels = torch.tensor([0] * 10 + [1] * 2)
    target = (labels[:, None] == labels[None, :]).float()[None]
    always_same = torch.full_like(class_imbalanced, 6.0)
    _check("always-same is penalised despite the positive majority", float(balanced_bce(always_same, target)) > 1.0)

    scene_id = torch.tensor([[0, 0, 1, 1]])
    logits = torch.where(scene_id[:, :, None] == scene_id[:, None, :], 5.0, -5.0)
    predictions = {
        "affinity_logits": logits,
        "affinity": torch.sigmoid(logits),
        "route_affinity_logits": logits,
        "route_affinity": torch.sigmoid(logits),
        "group_id": scene_id,
        "route_group_id": scene_id,
    }
    batch = {"affinity": (scene_id[:, :, None] == scene_id[:, None, :]).float(), "scene_id": scene_id}
    out = compute_affinity_loss(predictions, batch)
    _check("perfect prediction", float(out["loss_affinity"]) < 0.02 and float(out["affinity_f1"]) == 1.0)
    _check("partition metrics", float(out["ari"]) == 1.0 and float(out["exact_partition"]) == 1.0)
    _check("route metrics", "route_ari" in out and "route_f1" in out)


def test_pose_loss_ignores_the_gauge_of_each_scene() -> None:
    batch = _bag_batch()
    image_hw = batch["images"].shape[-2:]
    rotation = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    gauge = torch.eye(4)
    gauge[:3, :3] = rotation
    gauge[:3, 3] = torch.tensor([0.4, -0.2, 0.7])
    extrinsics = batch["extrinsics"]
    homogeneous = torch.eye(4).repeat(1, extrinsics.shape[1], 1, 1)
    homogeneous[..., :3, :] = extrinsics
    moved = (homogeneous @ gauge)[..., :3, :]
    shifted = extri_intri_to_pose_encoding(moved, batch["intrinsics"], image_hw)
    exact = extri_intri_to_pose_encoding(extrinsics, batch["intrinsics"], image_hw)

    criterion = MultitaskLoss(
        camera={"weight": 1.0, "weight_pairwise": 0.0, "normalize_trans_by_gt_scale": False},
    )
    moved_loss = criterion({"pose_enc_list": [shifted]}, dict(batch), 0.0)
    exact_loss = criterion({"pose_enc_list": [exact]}, dict(batch), 0.0)
    _check("a rigid gauge change costs nothing", float(moved_loss["loss_camera"]) < 1e-4)
    _check("exact poses cost nothing", float(exact_loss["loss_camera"]) < 1e-4)

    wrong = exact.clone()
    wrong[0, 1, 0] += 0.5
    _check("a real error is still penalised", float(criterion({"pose_enc_list": [wrong]}, dict(batch), 0.0)["loss_camera"]) > 1e-2)

    predictions, anchored = anchor_to_first_frame({"pose_enc_list": [exact[:, :3]]}, {**{k: v[:, :3] if torch.is_tensor(v) and v.ndim > 1 else v for k, v in batch.items()}})
    _check("first frame is identity", torch.allclose(anchored["extrinsics"][0, 0], torch.eye(3, 4), atol=1e-5))
    _check("prediction anchored too", torch.allclose(predictions["pose_enc_list"][0][0, 0, :3], torch.zeros(3), atol=1e-5))


def test_loss_splits_scenes() -> None:
    seen = []

    def loss_fn(predictions, batch):
        seen.append(predictions["pose_enc_list"][0].shape[1])
        return {"loss_camera": predictions["pose_enc_list"][0].sum()}

    scene_id = torch.tensor([[0, 0, 0, 1, 1]])
    pose = torch.zeros(1, 5, 9)
    mean_over_scenes(loss_fn, {"pose_enc_list": [pose]}, {"scene_id": scene_id}, scene_id)
    _check("scene sizes", seen == [3, 2])


def test_end_to_end_training_step() -> None:
    model = _tiny_model(gt_group_prob=[0.5, 0.5])
    model.train()
    batch = _bag_batch()
    criterion = _criterion()
    affinity_loss = MultitaskLoss(
        camera={"weight": 1.0, "weight_pairwise": 1.0, "normalize_trans_by_gt_scale": False},
        depth={"weight": 1.0, "min_valid_pts": 1, "use_conf_loss": False, "gradient_loss_config": None, "weight_normal": 0.0},
        point={"weight": 1.0, "min_valid_pts": 1},
        affinity={"weight": 1.0, "route_weight": 1.0, "include_output": True},
    )
    del criterion
    for _ in range(2):
        model.zero_grad()
        predictions = model(batch["images"], scene_id=batch["scene_id"])
        predictions["pose_enc_list"] = [predictions["pose_enc"]]
        losses = affinity_loss(predictions, batch, 0.7)
        losses["loss_objective"].backward()
    _check("finite objective", torch.isfinite(losses["loss_objective"]))
    for name in ("route_affinity", "affinity", "scene_attention"):
        grad = sum(float(p.grad.abs().sum()) for p in getattr(model.multiscene, name).parameters() if p.grad is not None)
        _check(f"{name} receives gradient", grad > 0.0)
    camera_grad = sum(float(p.grad.abs().sum()) for p in model.camera_head.parameters() if p.grad is not None)
    _check("camera head receives gradient", camera_grad > 0.0)
    _check("point loss reported", "loss_point" in losses and "loss_depth" in losses and "loss_camera" in losses)


# ---------------------------------------------------------------- evaluation


def test_eval_scores() -> None:
    scene_id = torch.tensor([0, 0, 1, 1])
    affinity = (scene_id[:, None] == scene_id[None, :]).float()
    prediction = {
        "affinity": affinity[None],
        "group_id": scene_id[None],
        "route_affinity": affinity[None],
        "route_group_id": scene_id[None],
    }
    scores = clustering_scores(prediction, scene_id)
    _check("perfect clustering", scores["affinity_f1"] == 1.0 and scores["ari"] == 1.0 and scores["exact_partition"] == 1.0)
    prediction["group_id"] = torch.zeros(1, 4, dtype=torch.long)
    _check("merged scenes are scored", clustering_scores(prediction, scene_id)["scene_count_error"] == 1.0)

    identity = torch.eye(3, 4).unsqueeze(0).repeat(2, 1, 1)
    turned = identity.clone()
    turned[1, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    same_scene = torch.zeros(2, dtype=torch.long)
    error = pose_scores(identity, turned, same_scene)
    _check("ninety degree rotation", error["rotation_deg"] > 80.0 and error["rra_15"] == 0.0)
    exact = pose_scores(identity, identity, same_scene)
    _check("matching poses", exact["rotation_deg"] < 1.0 and exact["rra_5"] == 1.0 and exact["auc_30"] > 0.99)

    target = torch.rand(4, 2, 2) + 1
    perfect = depth_scores(target, target, scene_id, min_valid=1)
    _check("matching depth", perfect["abs_rel"] < 1e-5 and perfect["delta125"] == 1.0)
    scaled = depth_scores(target * 3, target, scene_id, min_valid=1)
    _check("scale-aligned vs raw", scaled["abs_rel"] < 1e-5 and scaled["abs_rel_raw"] > 1.0)


def test_run_systems_on_a_model() -> None:
    model = _tiny_model()
    images = _tiny_bag()[0]
    scene_id = torch.tensor([0, 0, 0, 1, 1, 1])
    extrinsics = torch.eye(3, 4).repeat(6, 1, 1)
    extrinsics[:, 0, 3] = torch.linspace(0, 1, 6)
    depth = torch.rand(6, 32, 32) + 0.5
    report = run_systems(model, images, scene_id, extrinsics, depth, min_valid=1)
    for system in ("predicted", "oracle", "plain", "separate", "contamination", "cluster_gap"):
        _check(f"report has {system}", system in report)
    _check("oracle reports clustering", report["oracle"]["ari"] <= 1.0)


# ---------------------------------------------------------------- scene output


def test_scene_pack_uses_the_anchor() -> None:
    affinity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    _check("anchor is the best-connected frame", int(anchor_offset(affinity, torch.arange(3))) == 1)

    extrinsics = torch.eye(3, 4).unsqueeze(0).repeat(2, 1, 1)
    extrinsics[1, 0, 3] = 0.3
    relative = poses_relative_to_anchor(extrinsics, anchor=0)
    _check("anchor pose is identity", torch.allclose(relative[0], torch.eye(3, 4), atol=1e-5))
    _check("other pose keeps its translation", torch.allclose(relative[1, :, 3], torch.tensor([0.3, 0.0, 0.0])))

    depth = torch.full((1, 1, 1), 2.0)
    intrinsics = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]])
    points = points_in_anchor_frame(depth, torch.eye(3, 4).unsqueeze(0), intrinsics, anchor=0)
    _check("center point is the depth", torch.allclose(points[0, 0, 0], torch.tensor([0.0, 0.0, 2.0]), atol=1e-4))

    pose = torch.zeros(2, 9)
    pose[:, 6] = 1
    pose[:, 7:] = 0.5
    pose[1, 0] = 0.3
    scenes = pack_scenes(torch.eye(2), torch.zeros(2, dtype=torch.long), pose, torch.ones(2, 2, 2), (2, 2))
    _check("one scene", len(scenes) == 1 and scenes[0]["anchor"] == 0)
    _check("packed anchor pose", torch.allclose(scenes[0]["camera_poses"][0], torch.eye(3, 4), atol=1e-4))


def test_prediction_gap_reports_max() -> None:
    reference = {"pose_enc": torch.zeros(1, 2, 9), "depth": torch.ones(1, 2, 2, 2)}
    candidate = {"pose_enc": torch.zeros(1, 2, 9), "depth": torch.ones(1, 2, 2, 2)}
    candidate["pose_enc"][0, 0, 0] = 0.25
    gap = prediction_gap(reference, candidate)
    _check("gap max", abs(gap["pose_enc"]["max"] - 0.25) < 1e-6 and gap["depth"]["max"] == 0.0)


TESTS = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]


def main() -> None:
    for test in TESTS:
        print(f"-- {test.__name__}")
        test()
    print("multiscene smoke passed")


if __name__ == "__main__":
    main()
