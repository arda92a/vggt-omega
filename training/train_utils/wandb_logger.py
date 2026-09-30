# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import logging
from typing import Any, Mapping, Optional

from train_utils.distributed import get_machine_local_and_dist_rank


class WandbLogger:
    """Logs epoch averages from rank 0. A disabled logger never imports wandb."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "vggt-omega-multiscene",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        mode: str = "online",
        dir: Optional[str] = None,
    ) -> None:
        self._run = None
        if not enabled:
            return
        _, rank = get_machine_local_and_dist_rank()
        if rank != 0:
            return
        import wandb

        logging.info(f"Wandb run '{name}' in project '{project}' ({mode})")
        self._run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            mode=mode,
            dir=dir,
            id=name,
            resume="allow",
        )
        atexit.register(self.close)

    def log_epoch(self, metrics: Mapping[str, Any], epoch: int) -> None:
        if self._run is None:
            return
        payload = {key: value for key, value in metrics.items()}
        payload["epoch"] = epoch
        self._run.log(payload, step=epoch)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
