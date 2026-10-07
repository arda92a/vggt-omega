# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import logging
from typing import Any, Mapping, Optional

import numpy as np

from train_utils.distributed import get_machine_local_and_dist_rank

# Best value per metric, shown in the run summary and in the runs table.
_SUMMARY_BEST = {
    "val/predicted/ari": "max",
    "val/predicted/exact_partition": "max",
    "val/predicted/auc_30": "max",
    "val/predicted/rra_15": "max",
    "val/predicted/abs_rel": "min",
    "val/oracle/auc_30": "max",
    "val/oracle/abs_rel": "min",
}


class WandbLogger:
    """Logs from rank 0. A disabled logger never imports wandb.

    Three x-axes: `epoch` for `train/*` and `val/*`, `global_step` for `train_step/*`.
    """

    def __init__(
        self,
        enabled: bool = False,
        project: str = "vggt-omega-multiscene",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        mode: str = "online",
        dir: Optional[str] = None,
        step_freq: int = 10,
    ) -> None:
        self._run = None
        self._wandb = None
        self.step_freq = max(1, int(step_freq))
        if not enabled:
            return
        _, rank = get_machine_local_and_dist_rank()
        if rank != 0:
            return
        import wandb

        self._wandb = wandb
        logging.info(f"Wandb run '{name}' in project '{project}' ({mode})")
        init_kwargs = dict(project=project, name=name, entity=entity, mode=mode, dir=dir)
        try:
            # Same id resumes a crashed run. A deleted id cannot be reused.
            self._run = wandb.init(**init_kwargs, id=name, resume="allow")
        except Exception as exc:
            if "previously created and deleted" not in str(exc):
                raise
            logging.warning("Wandb run id %r was deleted. Starting a new run.", name)
            self._run = wandb.init(**init_kwargs)
        self._define_axes()
        atexit.register(self.close)

    def _define_axes(self) -> None:
        wandb = self._wandb
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("train_step/*", step_metric="global_step")
        for key, goal in _SUMMARY_BEST.items():
            wandb.define_metric(key, step_metric="epoch", summary=goal)

    def update_config(self, config: Mapping[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(dict(config), allow_val_change=True)

    def log_epoch(self, metrics: Mapping[str, Any], epoch: int) -> None:
        if self._run is None:
            return
        self._run.log({**metrics, "epoch": epoch})

    def log_step(self, metrics: Mapping[str, Any], step: int) -> None:
        if self._run is None or step % self.step_freq != 0:
            return
        self._run.log({**metrics, "global_step": step})

    def log_affinity(self, matrices: Mapping[str, np.ndarray], epoch: int) -> None:
        """Log affinity matrices of one validation bag side by side as one image."""
        if self._run is None:
            return
        tiles = [_heatmap(matrix) for matrix in matrices.values()]
        gap = np.full((tiles[0].shape[0], 4, 3), 255, dtype=np.uint8)
        row = np.concatenate([part for tile in tiles for part in (tile, gap)][:-1], axis=1)
        caption = " | ".join(matrices.keys())
        self._run.log({"val/affinity": self._wandb.Image(row, caption=caption), "epoch": epoch})

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def _heatmap(matrix: np.ndarray, size: int = 192) -> np.ndarray:
    """Grayscale image of a [0, 1] matrix, scaled up to roughly `size` pixels per side."""
    scale = max(1, size // max(matrix.shape))
    gray = (np.clip(matrix, 0.0, 1.0) * 255).astype(np.uint8)
    gray = np.kron(gray, np.ones((scale, scale), dtype=np.uint8))
    return np.repeat(gray[..., None], 3, axis=-1)
