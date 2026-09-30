"""CPU checks for the multi-scene pieces. No checkpoint and no dataset required.

Run from the training directory:

    python multiscene_smoke.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from check_single_scene import prediction_gap
from data.mixed_scene import MixedSceneCollate
from losses.base_loss.affinity import compute_affinity_loss
from losses.loss import MultitaskLoss
from losses.per_scene import mean_over_scenes
from multiscene_eval import clustering_scores, depth_scores, pose_scores, run_systems
from scene_pack import anchor_offset, pack_scenes, points_in_anchor_frame, poses_relative_to_anchor
from vggt_omega.models.multiscene import MultiScene, predict_pose_by_scene, scene_groups, select_camera_groups
from vggt_omega.models.vggt_omega import _build_multiscene
from vggt_omega.utils.pose_enc import encoding_to_camera


class _MeanCamera(torch.nn.Module):
    """Stand-in camera head: every frame receives the mean of the frames it was shown."""

    def forward(self, aggregated_tokens_list, patch_token_start):
        tokens = aggregated_tokens_list[-1]
        mixed = tokens[:, :, 0, :9].mean(dim=1, keepdim=True)
        return mixed.expand(tokens.shape[0], tokens.shape[1], 9).contiguous()


def _check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"ok  {name}")


def test_affinity_is_symmetric() -> None:
    head = MultiScene(dim=32, num_heads=4).affinity
    affinity = head(torch.randn(2, 5, 16, 32))
    _check("affinity diagonal", torch.allclose(affinity.diagonal(dim1=-2, dim2=-1), torch.ones(2, 5)))
    _check("affinity symmetric", torch.allclose(affinity, affinity.transpose(-1, -2)))
    _check("affinity range", bool(((affinity > 0) & (affinity <= 1)).all()))


def test_scene_attention_starts_as_identity() -> None:
    block = MultiScene(dim=32, num_heads=4, threshold=0.5)
    tokens = torch.randn(1, 6, 20, 32)
    scene = tokens[:, :, 1:17].clone()
    for hard_mask in (False, True):
        updated, _ = block(tokens, patch_token_start=17, hard_mask=hard_mask)
        _check(
            f"identity hard_mask={hard_mask}",
            torch.allclose(updated[:, :, 1:17], scene, atol=1e-5),
        )


def test_grouping() -> None:
    chained = torch.tensor(
        [[[1.0, 0.9, 0.1], [0.9, 1.0, 0.9], [0.1, 0.9, 1.0]]]
    )
    groups = scene_groups(chained, threshold=0.5)
    _check("chain is one group", torch.equal(groups, torch.zeros(1, 3, dtype=torch.long)))

    split = torch.tensor(
        [[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    groups = scene_groups(split, threshold=0.5)
    _check("two groups", torch.equal(groups, torch.tensor([[0, 0, 1]])))


def test_affinity_loss_falls() -> None:
    block = MultiScene(dim=16, num_heads=4)
    scene_tokens = torch.randn(4, 4, 16, 16)
    scene_id = torch.tensor([0, 0, 1, 1]).view(1, 4).expand(4, -1)
    target = (scene_id[:, :, None] == scene_id[:, None, :]).float()
    optimizer = torch.optim.Adam(block.affinity.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        affinity = block.affinity(scene_tokens)
        loss = compute_affinity_loss({"affinity": affinity}, {"affinity": target, "scene_id": scene_id.expand(4, -1)})
        loss["loss_affinity"].backward()
        optimizer.step()
        losses.append(float(loss["loss_affinity"]))
    _check("affinity loss fell", losses[-1] < losses[0])


def test_collate_packs_two_scenes() -> None:
    def scene(name: str) -> dict:
        return {
            "images": torch.rand(3, 3, 8, 8),
            "depths": torch.rand(3, 8, 8),
            "extrinsics": torch.eye(3, 4).expand(3, -1, -1).contiguous(),
            "intrinsics": torch.eye(3).expand(3, -1, -1).contiguous(),
            "world_points": torch.rand(3, 8, 8, 3),
            "point_masks": torch.ones(3, 8, 8, dtype=torch.bool),
            "seq_name": name,
            "dataset": "toy",
            "is_synthetic": False,
            "depth_train_mask": True,
        }

    batch = MixedSceneCollate(num_scenes=2, min_frames=3)([scene("a"), scene("b"), scene("c"), scene("d")])
    _check("packed batch", batch["images"].shape == (2, 6, 3, 8, 8))
    _check("scene ids", torch.equal(batch["scene_id"][0], torch.tensor([0, 0, 0, 1, 1, 1])))
    _check("affinity shape", batch["affinity"].shape == (2, 6, 6))
    _check("names joined", batch["seq_name"] == ["a|b", "c|d"])


def test_loss_uses_only_affinity() -> None:
    criterion = MultitaskLoss(affinity={"weight": 1.0, "include_output": False})
    affinity = torch.full((1, 2, 2), 0.5)
    affinity[:, range(2), range(2)] = 1
    batch = {
        "affinity": torch.eye(2).unsqueeze(0),
        "scene_id": torch.arange(2).view(1, 2),
    }
    output = criterion(
        {"affinity": affinity, "depth": torch.ones(1, 2, 4, 4), "pose_enc_list": [torch.zeros(1, 2, 9)]},
        batch,
        schedule_progress=0.0,
    )
    _check("objective is affinity", torch.allclose(output["loss_objective"], output["loss_affinity"]))


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
    _check("scene 0 receives grad", tokens.grad[0, :2].abs().sum().item() > 0)

    whole = torch.zeros(1, 4, dtype=torch.long)
    together = predict_pose_by_scene(_MeanCamera(), tokens.detach(), patch_token_start=1, group_id=whole)
    direct = _MeanCamera()([tokens.detach()], patch_token_start=1)
    _check("one group matches the plain head", torch.allclose(together, direct))


def test_camera_groups_follow_labels_only_when_asked() -> None:
    predicted = torch.tensor([[0, 0, 1, 1]])
    labels = torch.tensor([[1, 1, 0, 0]])
    chosen = select_camera_groups(predicted, labels, "gt")
    _check("gt groups", torch.equal(chosen, labels))
    chosen = select_camera_groups(predicted, None, "gt")
    _check("gt falls back at inference", torch.equal(chosen, predicted))
    chosen = select_camera_groups(predicted, labels, "predicted")
    _check("predicted groups", torch.equal(chosen, predicted))


def test_pose_loss_does_not_cross_scenes() -> None:
    seen = []

    def loss_fn(predictions, batch):
        frames = predictions["pose_enc_list"][0].shape[1]
        seen.append(frames)
        return {"loss_camera": predictions["pose_enc_list"][0].sum()}

    scene_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    pose = torch.arange(6, dtype=torch.float32).view(1, 6, 1).expand(1, 6, 9).contiguous()
    mean_over_scenes(loss_fn, {"pose_enc_list": [pose]}, {"scene_id": scene_id}, scene_id)
    _check("two scenes of three frames", seen == [3, 3])

    extrinsics = torch.eye(3, 4).view(1, 1, 3, 4).repeat(1, 4, 1, 1)
    extrinsics[0, 1, 0, 3] = 0.2
    extrinsics[0, 3, 0, 3] = 0.5
    pose = torch.zeros(1, 4, 9)
    pose[..., 3] = 1
    pose[..., 7:] = 1
    batch = {
        "scene_id": torch.tensor([[0, 0, 1, 1]]),
        "images": torch.rand(1, 4, 3, 8, 8),
        "extrinsics": extrinsics,
        "intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(1, 4, 1, 1),
        "valid_seq_mask": torch.ones(1, dtype=torch.bool),
        "depths": torch.ones(1, 4, 8, 8),
        "point_masks": torch.ones(1, 4, 8, 8, dtype=torch.bool),
        "depth_train_mask": torch.ones(1, dtype=torch.bool),
        "is_synthetic": torch.zeros(1, dtype=torch.bool),
        "affinity": torch.eye(4).unsqueeze(0),
    }
    criterion = MultitaskLoss(
        camera={"weight": 1.0, "weight_pairwise": 0.0, "normalize_trans_by_gt_scale": False},
        depth={
            "weight": 1.0,
            "min_valid_pts": 1,
            "use_conf_loss": False,
            "gradient_loss_config": None,
            "weight_normal": 0.0,
        },
    )
    output = criterion(
        {"pose_enc_list": [pose], "depth": torch.ones(1, 4, 8, 8), "affinity": batch["affinity"]},
        batch,
        schedule_progress=0.0,
    )
    _check("per-scene objective finite", torch.isfinite(output["loss_objective"]))
    _check("camera term present", "loss_camera" in output and "loss_depth" in output)


def test_scene_pack_uses_the_anchor() -> None:
    affinity = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    offset = anchor_offset(affinity, torch.arange(3))
    _check("anchor is the best-connected frame", int(offset) == 1)

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


def test_eval_scores_and_systems() -> None:
    affinity = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    scene_id = torch.tensor([0, 0, 1, 1])
    scores = clustering_scores(affinity, scene_id, threshold=0.5)
    _check("perfect groups", scores["affinity_f1"] == 1.0 and scores["group_agreement"] == 1.0)

    identity = torch.eye(3, 4).unsqueeze(0).repeat(2, 1, 1)
    turned = identity.clone()
    turned[1, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    error = pose_scores(identity, turned, torch.zeros(2, dtype=torch.long))
    _check("ninety degree rotation", error["rotation_deg"] > 80.0)
    same = pose_scores(identity, identity, torch.zeros(2, dtype=torch.long))
    _check("matching poses", same["rotation_deg"] < 1.0 and same["translation_deg"] < 1.0)

    signal = torch.tensor([0.0, 0.2, 1.0, 1.2])
    pose = torch.zeros(1, 4, 9)
    pose[0, :, 0] = signal
    pose[0, :, 6] = 1
    pose[0, :, 7:] = 0.5
    target_extrinsics = encoding_to_camera(pose, (2, 2))[0][0]
    target_depth = signal.view(4, 1, 1).expand(4, 2, 2).contiguous()
    images = torch.zeros(4, 3, 2, 2)
    images[:, 0, 0, 0] = signal

    class _BagModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.isolate_camera = True

        def forward(self, images, scene_id=None):
            if images.ndim == 4:
                images = images.unsqueeze(0)
            frames = images.shape[1]
            output = images.new_zeros(1, frames, 9)
            output[..., 6] = 1
            output[..., 7:] = 0.5
            if frames == 4:
                depth = images.new_full((1, frames, 2, 2), 5.0)
            else:
                output[..., 0] = images[:, :, 0, 0, 0]
                depth = images[:, :, 0, 0, 0].view(1, frames, 1, 1).expand(1, frames, 2, 2).contiguous()
            affinity = torch.eye(frames, device=images.device, dtype=images.dtype).unsqueeze(0)
            return {
                "pose_enc": output,
                "depth": depth,
                "affinity": affinity,
                "group_id": torch.zeros(1, frames, dtype=torch.long),
            }

    report = run_systems(
        _BagModel(),
        images,
        scene_id,
        target_extrinsics,
        target_depth,
        min_valid=1,
    )
    _check(
        "oracle beats the mixed bag",
        report["oracle"]["translation_deg"] < report["plain"]["translation_deg"]
        and report["oracle"]["abs_rel"] < report["plain"]["abs_rel"],
    )
    _check(
        "contamination is the gap",
        abs(
            report["contamination"]["translation_deg"]
            - (report["plain"]["translation_deg"] - report["oracle"]["translation_deg"])
        )
        < 1e-5,
    )
    depth = depth_scores(target_depth, target_depth, scene_id, min_valid=1)
    _check("matching depth", depth["abs_rel"] < 1e-5)


def test_prediction_gap_reports_max() -> None:
    reference = {"pose_enc": torch.zeros(1, 2, 9), "depth": torch.ones(1, 2, 2, 2)}
    candidate = {"pose_enc": torch.zeros(1, 2, 9), "depth": torch.ones(1, 2, 2, 2)}
    candidate["pose_enc"][0, 0, 0] = 0.25
    gap = prediction_gap(reference, candidate)
    _check("gap max", abs(gap["pose_enc"]["max"] - 0.25) < 1e-6 and gap["depth"]["max"] == 0.0)


def test_disabled_block_is_absent() -> None:
    _check("unset", _build_multiscene(None, dim=32) is None)
    _check("disabled", _build_multiscene({"enabled": False}, dim=32) is None)
    built = _build_multiscene({"enabled": True, "num_heads": 4, "threshold": 0.4}, dim=32)
    _check("enabled", isinstance(built, MultiScene) and built.threshold == 0.4)


def main() -> None:
    test_affinity_is_symmetric()
    test_scene_attention_starts_as_identity()
    test_grouping()
    test_affinity_loss_falls()
    test_collate_packs_two_scenes()
    test_loss_uses_only_affinity()
    test_camera_head_stays_inside_its_scene()
    test_camera_groups_follow_labels_only_when_asked()
    test_pose_loss_does_not_cross_scenes()
    test_scene_pack_uses_the_anchor()
    test_eval_scores_and_systems()
    test_prediction_gap_reports_max()
    test_disabled_block_is_absent()
    print("multiscene smoke passed")


if __name__ == "__main__":
    main()
