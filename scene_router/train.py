"""Train a scene router on the Stitch_Requests bags. Two models (config key `model`):

    dino_router  DINOv3 + pooling + bag transformer               scene_router/config.yaml
    vggt_match   frozen VGGT-Omega block 4 + learned patch matching  scene_router/config_vggt_match.yaml

    uv run python scene_router/make_bags.py                                              # once: train/val bags
    uv run python -m scene_router.train --config scene_router/config.yaml                 # DINOv3 router
    uv run python -m scene_router.train --config scene_router/config_vggt_match.yaml      # VGGT patch matching

Every epoch logs loss, AUROC, ARI and perfect-bag ratio for train and val (see metrics.py), to the console,
to <out>/train.log and to wandb when wandb=true, with the same names for both models. best/* in wandb is the
best epoch so far (by val ARI), logged every epoch and kept in the run summary. <out>/best.pt keeps that epoch.
"""

import argparse
import contextlib
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from scene_router.data import BagDataset, collate
from scene_router.metrics import EpochMetrics, affinity_loss
from scene_router.model import build_model

log = logging.getLogger("scene_router")


# ---------------------------------------------------------------- setup


def load_config() -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scene_router/config.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value pairs that replace config values")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.overrides:
        key, _, value = item.partition("=")
        if key not in cfg:
            raise SystemExit(f"unknown config key {key!r}, known: {', '.join(cfg)}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(out / "train.log")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data = Path(cfg["data_dir"])
    resize = cfg.get("resize", "stretch")
    train_set = BagDataset(data / "train.json", cfg["height"], cfg["width"], augment=True, resize=resize)
    val_set = BagDataset(data / "val.json", cfg["height"], cfg["width"], resize=resize)
    per_epoch = min(cfg["epoch_bags"] or len(train_set), len(train_set))
    common = dict(batch_size=cfg["bags_per_batch"], num_workers=cfg["workers"], collate_fn=collate,
                  persistent_workers=cfg["workers"] > 0)
    train = DataLoader(train_set, sampler=RandomSampler(train_set, num_samples=per_epoch), drop_last=True, **common)
    val = DataLoader(val_set, shuffle=False, **common)
    return train, val


def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    if hasattr(model, "param_groups"):  # the model knows its own groups (vggt_match)
        return torch.optim.AdamW(model.param_groups(cfg))
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    head = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("backbone.")]
    groups = [{"params": head, "lr": cfg["lr"]}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg["backbone_lr"]})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


# ---------------------------------------------------------------- loops


def run_epoch(model, loader, device, cfg, autocast, optimizer=None, scheduler=None, desc="") -> dict:
    """One pass over loader. Trains when an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    metrics = EpochMetrics(cfg["threshold"])
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.set_grad_enabled(training):
        for batch in bar:
            images, labels, mask = batch["images"].to(device), batch["labels"].to(device), batch["mask"].to(device)
            with autocast():
                logits = model(images, mask)
            loss = affinity_loss(logits.float(), labels, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
            metrics.add(loss, logits, labels, mask)
            bar.set_postfix(loss=f"{loss.item():.3f}")
    return metrics.compute()


def format_metrics(values: dict) -> str:
    return "  ".join(f"{name} {value:.3f}" for name, value in values.items())


def save(model: torch.nn.Module, cfg: dict, epoch: int, metrics: dict, path: Path) -> None:
    state = model.checkpoint_state() if hasattr(model, "checkpoint_state") else model.state_dict()
    torch.save({"model": state, "config": cfg, "epoch": epoch, "metrics": metrics}, path)


# ---------------------------------------------------------------- main


def main() -> None:
    cfg = load_config()
    out = Path(cfg["out"])
    setup_logging(out)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed_everything(cfg["seed"])
    device = pick_device(cfg["device"])

    train_loader, val_loader = make_loaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, cfg["warmup_steps"], total_steps))
    use_amp = cfg["amp"] and device.type == "cuda"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"device {device}  |  {cfg.get('model', 'dino_router')}: {model.describe()}  |  "
             f"trainable {trainable / 1e6:.1f}M of {total / 1e6:.1f}M parameters")
    log.info(f"train {len(train_loader.dataset)} bags ({len(train_loader.sampler)} per epoch, "
             f"{len(train_loader)} steps)  |  val {len(val_loader.dataset)} bags  |  output {out}")

    run = None
    if cfg["wandb"]:
        import wandb

        run = wandb.init(project=cfg["wandb_project"], name=cfg["wandb_name"], config=cfg, dir=str(out))

        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")  # every chart uses epoch as its x axis

    best_ari, best, history = -1.0, {}, []
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        tag = f"epoch {epoch}/{cfg['epochs']}"
        lr = scheduler.get_last_lr()[0]
        train_metrics = run_epoch(model, train_loader, device, cfg, autocast, optimizer, scheduler, desc=f"{tag} train")
        val_metrics = run_epoch(model, val_loader, device, cfg, autocast, desc=f"{tag} val")
        minutes = (time.time() - start) / 60

        log.info(f"{tag}  ({minutes:.1f} min)")
        log.info(f"  train  {format_metrics(train_metrics)}")
        log.info(f"  val    {format_metrics(val_metrics)}")

        save(model, cfg, epoch, val_metrics, out / "last.pt")
        if val_metrics["ari"] > best_ari:
            best_ari = val_metrics["ari"]
            best = {"best/epoch": epoch, **{f"best/{k}": v for k, v in val_metrics.items()}}
            save(model, cfg, epoch, val_metrics, out / "best.pt")
            log.info(f"  new best val ARI {best_ari:.3f}, saved {out / 'best.pt'}")
        if hasattr(model, "describe") and cfg.get("model") == "vggt_match":
            log.info(f"  head   {model.describe()}")

        # best/* is the best epoch so far (val metrics of best.pt), so runs can be compared on one chart.
        record = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()},
                  **{f"val/{k}": v for k, v in val_metrics.items()}, **best}
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        if run is not None:
            run.log({**record, "lr": lr})
            run.summary.update(best)

    log.info(f"done. best epoch {best['best/epoch']}  |  " + format_metrics(
        {k.removeprefix("best/"): v for k, v in best.items() if k != "best/epoch"}))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
