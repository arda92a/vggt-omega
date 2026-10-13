# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from trainer import Trainer


def free_run_name(name: str, root: str = "logs") -> str:
    """`name` if logs/name does not exist yet, else name_2, name_3, ... so no run reuses an old one."""
    candidate, number = name, 2
    while (Path(root) / candidate).exists():
        candidate = f"{name}_{number}"
        number += 1
    return candidate


def load_config(config_name: str = "default", overrides=None, new_run: bool = False):
    """Compose the config exactly as training does, so callers can inspect it without training.

    With `new_run`, exp_name is replaced by the first free name, which moves the log directory,
    the checkpoints and the wandb run name with it.
    """
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))

    if new_run:
        cfg.exp_name = free_run_name(str(cfg.exp_name))
    # Resolve ${...} in place but keep a DictConfig: the trainer reads nested config with
    # attribute access, which a plain dict does not support.
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    # exp_name only exists so other paths can interpolate ${exp_name}; it is not a Trainer argument.
    cfg.pop("exp_name", None)
    return cfg


def run(config_name: str = "default", overrides=None) -> None:
    Trainer(**load_config(config_name, overrides, new_run=True)).run()


def main():
    parser = argparse.ArgumentParser(description="Train VGGT-Omega")
    parser.add_argument("--config", default="default", help="config name under training/config, without .yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="hydra-style overrides, e.g. max_epochs=1 optim.optimizer.lr=1e-5",
    )
    args = parser.parse_args()

    run(args.config, args.overrides)


if __name__ == "__main__":
    main()
