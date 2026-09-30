"""Pack several single-scene samples into one mixed-scene training bag.

Each dataset sample is already one scene, normalized to its own first camera.
This collate concatenates S such samples along the frame axis and records which
frames share a scene. Geometry stays in each scene's own frame; do not run the
global GT normalization on the packed bag.
"""

from typing import Sequence

import torch


class MixedSceneCollate:
    """Collate that builds batches of `num_scenes` scenes and `frames` frames each."""

    def __init__(self, num_scenes: int = 2, min_frames: int = 3) -> None:
        if num_scenes < 2:
            raise ValueError(f"num_scenes must be at least 2, got {num_scenes}")
        if min_frames < 1:
            raise ValueError(f"min_frames must be positive, got {min_frames}")
        self.num_scenes = int(num_scenes)
        self.min_frames = int(min_frames)

    def __call__(self, samples: Sequence[dict]) -> dict:
        if len(samples) < self.num_scenes:
            raise ValueError(
                f"Need at least {self.num_scenes} scenes to build a mixed batch, got {len(samples)}. "
                "Raise data.train.per_gpu_batch_scale so the loader batch is a multiple of num_scenes."
            )

        usable = len(samples) - (len(samples) % self.num_scenes)
        packed = [
            _pack_scenes(samples[start : start + self.num_scenes], self.min_frames)
            for start in range(0, usable, self.num_scenes)
        ]
        return _stack_packed(packed)


def _pack_scenes(scenes: Sequence[dict], min_frames: int) -> dict:
    frame_counts = [int(scene["images"].shape[0]) for scene in scenes]
    if len(set(frame_counts)) != 1:
        raise ValueError(f"Every scene in a bag must have the same frame count, got {frame_counts}")
    num_frames = frame_counts[0]
    if num_frames < min_frames:
        raise ValueError(f"Each scene must have at least {min_frames} frames, got {num_frames}")

    shapes = [tuple(scene["images"].shape[-2:]) for scene in scenes]
    if len(set(shapes)) != 1:
        raise ValueError(f"Every scene in a bag must have the same image size, got {shapes}")

    packed = {}
    for key in ("images", "depths", "extrinsics", "intrinsics", "world_points", "point_masks"):
        packed[key] = torch.cat([scene[key] for scene in scenes], dim=0)

    for key in ("tracks", "track_vis_mask"):
        if key in scenes[0]:
            packed[key] = torch.cat([scene[key] for scene in scenes], dim=0)

    scene_id = torch.cat(
        [
            torch.full((num_frames,), scene_index, dtype=torch.long)
            for scene_index in range(len(scenes))
        ]
    )
    packed["scene_id"] = scene_id
    packed["affinity"] = (scene_id[:, None] == scene_id[None, :]).to(dtype=torch.float32)
    packed["seq_name"] = "|".join(str(scene["seq_name"]) for scene in scenes)
    packed["dataset"] = "|".join(str(scene["dataset"]) for scene in scenes)
    packed["is_synthetic"] = all(bool(scene.get("is_synthetic", False)) for scene in scenes)
    packed["depth_train_mask"] = all(bool(scene.get("depth_train_mask", True)) for scene in scenes)
    return packed


def _stack_packed(packed: Sequence[dict]) -> dict:
    batch = {}
    tensor_keys = [key for key, value in packed[0].items() if torch.is_tensor(value)]
    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in packed], dim=0)
    batch["seq_name"] = [sample["seq_name"] for sample in packed]
    batch["dataset"] = [sample["dataset"] for sample in packed]
    batch["is_synthetic"] = torch.tensor([sample["is_synthetic"] for sample in packed], dtype=torch.bool)
    batch["depth_train_mask"] = torch.tensor(
        [sample["depth_train_mask"] for sample in packed],
        dtype=torch.bool,
    )
    batch["valid_seq_mask"] = torch.ones(len(packed), dtype=torch.bool)
    return batch
