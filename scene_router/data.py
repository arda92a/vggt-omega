"""Bags from make_bags.py as training samples: the frames of one bag and the sequence label of every frame."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


def load_frame(path: Path, height: int, width: int) -> torch.Tensor:
    """The whole frame resized to (height, width), no crop: uint8 tensor (3, H, W)."""
    with Image.open(path) as image:
        image.draft("RGB", (width, height))  # JPEG decodes at a smaller scale directly, much faster for big photos
        image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


class BagDataset(Dataset):
    def __init__(self, bags_file: str | Path, height: int, width: int, augment: bool = False):
        content = json.loads(Path(bags_file).read_text())
        self.root = Path(content["root"])
        self.bags = content["bags"]
        self.height, self.width = height, width
        # Mild colour changes only: geometry stays as it is, so overlaps between neighbouring frames stay real.
        self.augment = v2.ColorJitter(0.2, 0.2, 0.2, 0.02) if augment else None

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int) -> dict:
        bag = self.bags[index]
        frames = [load_frame(self.root / f, self.height, self.width) for f in bag["frames"]]
        if self.augment is not None:
            frames = [self.augment(frame) for frame in frames]
        return {
            "images": torch.stack(frames),  # (N, 3, H, W) uint8
            "labels": torch.tensor(bag["labels"]),  # (N,) sequence index inside the bag
            "id": bag["id"],
        }


def collate(items: list[dict]) -> dict:
    """Pads bags to the largest one. mask is True for real frames, labels are -1 on padding."""
    size = max(len(item["labels"]) for item in items)
    _, channels, height, width = items[0]["images"].shape
    images = torch.zeros(len(items), size, channels, height, width, dtype=torch.uint8)
    labels = torch.full((len(items), size), -1, dtype=torch.long)
    for i, item in enumerate(items):
        n = len(item["labels"])
        images[i, :n] = item["images"]
        labels[i, :n] = item["labels"]
    return {"images": images, "labels": labels, "mask": labels >= 0, "ids": [item["id"] for item in items]}
