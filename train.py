"""Train multi-scene VGGT-Omega.

    uv run python train.py                                   # data in ./MERGED_DIR
    uv run python train.py --data-dir /path/to/merged
    uv run python train.py --config multiscene_stage2 checkpoint.model_weight_path=...
    uv run torchrun --nproc_per_node=8 train.py ...          # several GPUs
    uv run python train.py --eval-only checkpoint.model_weight_path=...

Anything after the options is a hydra override, for example `max_epochs=1`. The dataset root is
--data-dir, else the MERGED_DIR environment variable, else the MERGED_DIR folder in the repository.
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))


def _use_single_process_defaults() -> None:
    """Fill in the variables torchrun would set, so a plain `python train.py` runs on one GPU."""
    if "RANK" in os.environ:
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    for name, value in {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
    }.items():
        os.environ.setdefault(name, value)


def _find_data_dir(explicit):
    """--data-dir, else $MERGED_DIR, else a MERGED_DIR folder in the working directory or the repository."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise SystemExit(f"--data-dir {explicit} is not a directory")
        return path.resolve()
    for candidate in (os.environ.get("MERGED_DIR"), "MERGED_DIR", ROOT / "MERGED_DIR"):
        if candidate and Path(candidate).expanduser().is_dir():
            return Path(candidate).expanduser().resolve()
    return None


def _check_dataset_roots(cfg) -> None:
    missing = []
    for name, dataset in cfg.data.train.dataset.dataset_configs.items():
        if dataset is None:
            continue
        root = dataset.get("UNIFIED_DIR")
        if not root or not Path(str(root)).is_dir():
            missing.append(f"  {name}: {root}")
    if missing:
        raise SystemExit(
            "Dataset directories not found:\n"
            + "\n".join(missing)
            + "\nPass --data-dir, set MERGED_DIR, or put the data in a MERGED_DIR folder."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-scene VGGT-Omega")
    parser.add_argument("--config", default="multiscene_stage1", help="config name under training/config")
    parser.add_argument(
        "--data-dir",
        help="unified dataset root with train.txt and val.txt (default: $MERGED_DIR, else ./MERGED_DIR)",
    )
    parser.add_argument("--eval-only", action="store_true", help="run one validation pass and exit")
    parser.add_argument("overrides", nargs="*", help="hydra overrides, e.g. max_epochs=1")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Training needs a CUDA GPU. Run this on the GPU machine.")

    from launch import load_config
    from trainer import Trainer

    overrides = list(args.overrides)
    data_dir = _find_data_dir(args.data_dir)
    if data_dir is not None:
        overrides.append(f"data.train.dataset.dataset_configs.merged.UNIFIED_DIR={json.dumps(str(data_dir))}")
    if args.eval_only:
        overrides.append("eval_only=true")
    cfg = load_config(args.config, overrides)
    _check_dataset_roots(cfg)

    _use_single_process_defaults()
    Trainer(**cfg).run()


if __name__ == "__main__":
    main()
