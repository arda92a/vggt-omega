"""Pack several single-scene samples into one mixed-scene training bag.

Each dataset sample is one scene, already normalized to its own first camera. The collate
draws a scene count S, picks S samples from distinct sequences, crops every scene to a
random frame count, concatenates them along the frame axis, and shuffles the frames.
It returns one bag per call (batch size 1). Geometry stays in each scene's own frame, so
do not run the global GT normalization on the packed bag.

S = 1 is allowed on purpose: the model must also leave a single scene alone.
"""

import random
from collections import defaultdict
from typing import Optional, Sequence

import torch


_FRAME_KEYS = ("images", "depths", "extrinsics", "intrinsics", "world_points", "point_masks")
_OPTIONAL_FRAME_KEYS = ("tracks", "track_vis_mask")


class MixedSceneCollate:
    """Collate that builds one bag of S scenes with a random number of frames per scene.

    Args:
        num_scenes_range: [min, max] scenes per bag. `num_scenes` is a shortcut for [n, n].
        scene_count_weights: sampling weight for each count in the range. Uniform if unset.
        min_frames: minimum frames kept per scene.
        frames_range: optional [min, max] frames kept per scene. Defaults to all of them.
        max_total_frames: optional cap on the bag size. Scenes are cropped, then S is lowered.
        shuffle_frames: shuffle the frame order, so position carries no scene information.
        same_dataset_prob: chance of drawing every scene from one dataset (harder negatives).
    """

    def __init__(
        self,
        num_scenes: Optional[int] = None,
        num_scenes_range: Optional[Sequence[int]] = None,
        scene_count_weights: Optional[Sequence[float]] = None,
        min_frames: int = 3,
        frames_range: Optional[Sequence[int]] = None,
        max_total_frames: Optional[int] = None,
        shuffle_frames: bool = True,
        same_dataset_prob: float = 0.0,
    ) -> None:
        if num_scenes is not None:
            num_scenes_range = (num_scenes, num_scenes)
        low, high = (int(value) for value in (num_scenes_range or (2, 2)))
        if low < 1 or high < low:
            raise ValueError(f"num_scenes_range must satisfy 1 <= min <= max, got {[low, high]}")
        if min_frames < 1:
            raise ValueError(f"min_frames must be positive, got {min_frames}")
        if scene_count_weights is not None and len(scene_count_weights) != high - low + 1:
            raise ValueError(
                f"scene_count_weights needs {high - low + 1} entries for range {[low, high]}, "
                f"got {len(scene_count_weights)}"
            )
        if max_total_frames is not None and max_total_frames < min_frames:
            raise ValueError(f"max_total_frames {max_total_frames} is below min_frames {min_frames}")

        self.scene_counts = list(range(low, high + 1))
        self.scene_count_weights = [float(w) for w in scene_count_weights] if scene_count_weights else None
        self.min_frames = int(min_frames)
        self.frames_range = tuple(int(v) for v in frames_range) if frames_range else None
        self.max_total_frames = int(max_total_frames) if max_total_frames else None
        self.shuffle_frames = bool(shuffle_frames)
        self.same_dataset_prob = float(same_dataset_prob)

    def __call__(self, samples: Sequence[dict]) -> dict:
        samples = _distinct_sequences(samples)
        num_scenes = self._draw_scene_count(len(samples))
        chosen = self._choose(samples, num_scenes)
        scenes = self._crop(chosen)
        return _batch_of_one(_pack(scenes, self.shuffle_frames))

    def _draw_scene_count(self, available: int) -> int:
        limit = available
        if self.max_total_frames is not None:
            limit = min(limit, self.max_total_frames // self.min_frames)
        pairs = [
            (count, 1.0 if self.scene_count_weights is None else self.scene_count_weights[index])
            for index, count in enumerate(self.scene_counts)
            if count <= limit
        ]
        if not pairs:
            raise ValueError(
                f"Need at least {self.scene_counts[0]} distinct sequences per loader batch, got {available}. "
                "Raise data.train.per_gpu_batch_scale."
            )
        counts, weights = zip(*pairs)
        return random.choices(counts, weights=weights)[0]

    def _choose(self, samples: Sequence[dict], num_scenes: int) -> list[dict]:
        if num_scenes > 1 and random.random() < self.same_dataset_prob:
            by_dataset = defaultdict(list)
            for sample in samples:
                by_dataset[str(sample["dataset"])].append(sample)
            groups = [group for group in by_dataset.values() if len(group) >= num_scenes]
            if groups:
                return random.sample(random.choice(groups), num_scenes)
        return random.sample(list(samples), num_scenes)

    def _crop(self, scenes: Sequence[dict]) -> list[dict]:
        available = [int(scene["images"].shape[0]) for scene in scenes]
        if min(available) < self.min_frames:
            raise ValueError(f"Each scene needs at least {self.min_frames} frames, got {available}")
        counts = list(available)
        if self.frames_range is not None:
            low, high = max(self.min_frames, self.frames_range[0]), max(self.min_frames, self.frames_range[1])
            counts = [random.randint(min(low, count), min(high, count)) for count in available]
        if self.max_total_frames is not None:
            while sum(counts) > self.max_total_frames:
                widest = max(range(len(counts)), key=counts.__getitem__)
                counts[widest] -= 1
        return [_crop_scene(scene, count) for scene, count in zip(scenes, counts)]


def _distinct_sequences(samples: Sequence[dict]) -> list[dict]:
    """Two samples of one sequence would be labelled as different scenes. Keep the first."""
    seen, kept = set(), []
    for sample in samples:
        key = (str(sample["dataset"]), str(sample["seq_name"]))
        if key not in seen:
            seen.add(key)
            kept.append(sample)
    return kept


def _crop_scene(scene: dict, count: int) -> dict:
    total = int(scene["images"].shape[0])
    if count >= total:
        return scene
    keep = torch.tensor(sorted(random.sample(range(total), count)), dtype=torch.long)
    cropped = dict(scene)
    for key in (*_FRAME_KEYS, *_OPTIONAL_FRAME_KEYS):
        if key in scene:
            cropped[key] = scene[key].index_select(0, keep)
    return cropped


def _pack(scenes: Sequence[dict], shuffle: bool) -> dict:
    shapes = [tuple(scene["images"].shape[-2:]) for scene in scenes]
    if len(set(shapes)) != 1:
        raise ValueError(f"Every scene in a bag must have the same image size, got {shapes}")

    packed = {}
    for key in (*_FRAME_KEYS, *_OPTIONAL_FRAME_KEYS):
        if key in scenes[0]:
            packed[key] = torch.cat([scene[key] for scene in scenes], dim=0)
    scene_id = torch.cat(
        [torch.full((int(scene["images"].shape[0]),), index, dtype=torch.long) for index, scene in enumerate(scenes)]
    )
    if shuffle:
        order = torch.randperm(scene_id.shape[0])
        packed = {key: value[order] for key, value in packed.items()}
        scene_id = scene_id[order]

    packed["scene_id"] = scene_id
    packed["affinity"] = (scene_id[:, None] == scene_id[None, :]).to(dtype=torch.float32)
    packed["num_scenes"] = torch.tensor(float(len(scenes)))
    packed["num_frames"] = torch.tensor(float(scene_id.shape[0]))
    packed["seq_name"] = "|".join(str(scene["seq_name"]) for scene in scenes)
    packed["dataset"] = "|".join(str(scene["dataset"]) for scene in scenes)
    packed["is_synthetic"] = all(bool(scene.get("is_synthetic", False)) for scene in scenes)
    packed["depth_train_mask"] = all(bool(scene.get("depth_train_mask", True)) for scene in scenes)
    return packed


def _batch_of_one(packed: dict) -> dict:
    batch = {key: value.unsqueeze(0) for key, value in packed.items() if torch.is_tensor(value)}
    batch["seq_name"] = [packed["seq_name"]]
    batch["dataset"] = [packed["dataset"]]
    batch["is_synthetic"] = torch.tensor([packed["is_synthetic"]], dtype=torch.bool)
    batch["depth_train_mask"] = torch.tensor([packed["depth_train_mask"]], dtype=torch.bool)
    batch["valid_seq_mask"] = torch.ones(1, dtype=torch.bool)
    return batch
