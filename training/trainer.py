# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# Reduces fragmentation in the caching allocator, which matters at these activation sizes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Surfaces NCCL errors instead of hanging.
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import contextlib
import gc
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf


from train_utils.checkpoint import (
    DDPCheckpointSaver,
    load_dinov3_weights,
    load_model_weights,
)
from train_utils.console import Progress
from train_utils.distributed import (
    get_machine_local_and_dist_rank,
    setup_distributed_backend,
    unwrap_ddp_if_wrapped,
)
from train_utils.freeze import freeze_modules, freeze_parameters
from train_utils.fsdp import (
    ShardedCheckpointSaver,
    apply_fsdp2_strategy,
    create_grad_scaler,
    to_fsdp_settings,
)
from train_utils.general import (
    AverageMeter,
    DurationMeter,
    copy_data_to_device,
    get_amp_type,
    makedir,
    set_seeds,
)
from train_utils import report
from train_utils.logging import setup_logging
from train_utils.optimizer import construct_optimizers
from trainer_conf import (
    TrainerCheckpointConf,
    TrainerCudaConf,
    TrainerDistributedConf,
    TrainerLoggingConf,
    TrainerOptimConf,
)
from trainer_utils import chunk_batch_for_accum_steps


class Trainer:
    """
    Trainer supporting the DDP and FSDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        seed_value: int = 123,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        strategy: str = "ddp",
        fsdp_settings: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        start_epoch: int = 0,
        validation: Optional[Dict[str, Any]] = None,
        eval_only: bool = False,
    ):
        self.accum_steps = accum_steps
        self.validation_conf = validation or {}
        self.eval_only = eval_only
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = TrainerLoggingConf(**logging)
        self.checkpoint_conf = TrainerCheckpointConf(**checkpoint)
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches
        self.optim_conf_raw = optim
        self.optim_conf = TrainerOptimConf(**optim or {})
        self.loss_conf = loss
        distributed = TrainerDistributedConf(**distributed or {})
        cuda = TrainerCudaConf(**cuda or {})
        self.schedule_progress = 0.0
        self.seed_value = seed_value

        self._infer_distributed_backend_if_none(distributed)
        self.strategy = strategy
        self.fsdp_settings = to_fsdp_settings(fsdp_settings)
        self.start_epoch = start_epoch
        self._check_strategy_consistency()

        self._setup_device()

        self._setup_torch_dist_and_backend(cuda, distributed)

        setup_logging(
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        # TODO: Enable separate seed setting for each data worker.
        # Rank 0's seed, deliberately, because the model is built under it: set_seeds offsets
        # by rank, so seeding per rank here would have every rank draw a *different* random
        # init. DDP hides that by broadcasting rank 0's parameters at construction, but
        # fully_shard does not broadcast, so FSDP would shard a patchwork of per-rank draws.
        # run_train reseeds with the rank offset before every epoch, which is what actually
        # gives each rank its own data order.
        set_seeds(seed_value, self.max_epochs, dist_rank=0)
        self._setup_components()  # Except Optimizer everything is setup here.
        self._setup_dataloaders()
        self._move_to_device()

        self.time_elapsed_meter = DurationMeter("Time Elapsed")

        if self._is_fsdp_training():
            self._setup_fsdp()

        else:
            self._construct_optimizers()
            self._load_initial_model_weights()
            self._setup_ddp_distributed_training(distributed)

        if self.gradient_clipper is not None:
            self.gradient_clipper.setup_clipping(self.model)
        dist.barrier()

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.est_epoch_time = {"train": 0}
    def _infer_distributed_backend_if_none(self, distributed_conf):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
        torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
        torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32
        setup_distributed_backend(distributed_conf.backend, distributed_conf.timeout_mins)

    def _setup_device(self):
        self.local_rank, self.rank = get_machine_local_and_dist_rank()
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.local_rank)

    def _is_fsdp_training(self) -> bool:
        return self.strategy == "fsdp"

    def _check_strategy_consistency(self):
        if self.strategy not in {"ddp", "fsdp"}:
            raise ValueError(f"Unsupported training strategy: {self.strategy}")
        if self.accum_steps < 1:
            raise ValueError(f"accum_steps must be positive, got {self.accum_steps}")
        error_msg = "FSDP settings should be set if and only if FSDP training strategy is selected"
        has_fsdp_settings = self.fsdp_settings is not None
        assert self._is_fsdp_training() == has_fsdp_settings, error_msg
        assert not (
            self._is_fsdp_training() and self.accum_steps > 1
        ), "FSDP2 with accum_steps > 1 is not currently supported"

        if self._is_fsdp_training():
            fsdp_param_dtype = self.fsdp_settings.param_dtype
            if self.optim_conf.amp.enabled:
                amp_dtype = self.optim_conf.amp.amp_dtype
                if fsdp_param_dtype is not None and amp_dtype != fsdp_param_dtype:
                    raise ValueError(
                        "AMP dtype must match FSDP param dtype. "
                        f"Got amp_dtype={amp_dtype}, fsdp_param_dtype={fsdp_param_dtype}"
                    )
            elif fsdp_param_dtype in {"bfloat16", "float16"}:
                raise ValueError(
                    "AMP must be enabled when FSDP parameters use a reduced-precision dtype"
                )

    def _setup_ddp_distributed_training(self, distributed_conf):
        assert not self._is_fsdp_training()
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            **ddp_options,
        )

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )
        self.model.to(self.device)

        if self.loss:
            copy_data_to_device(self.loss, self.device)
        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
        }
        if self._is_fsdp_training():
            saver = ShardedCheckpointSaver(
                checkpoint_folder,
                checkpoint_names=checkpoint_names,
                rank=self.rank,
            )
            # For FSDP2/DCP, pass optimizer objects so they can be sharded-saved
            actual_optimizers = [optim.optimizer for optim in self.optims]

            saver.save_checkpoint(
                model=self.model,
                optimizers=actual_optimizers,
                **checkpoint_content,
            )
        else:
            saver = DDPCheckpointSaver(
                checkpoint_folder,
                checkpoint_names=checkpoint_names,
                rank=self.rank,
                epoch=epoch,
            )
            saver.save_checkpoint(
                model=unwrap_ddp_if_wrapped(self.model),
                **checkpoint_content,
            )

    def _get_scalar_log_keys(self, phase):
        if self.logging_conf.scalar_keys_to_log is not None:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        else:
            return []


    def _model_to_fsdp(self):
        # Apply FSDP2 Strategy (includes wrapping)
        self.model = apply_fsdp2_strategy(self.model, self.fsdp_settings)

    def _setup_fsdp(self):
        assert self._is_fsdp_training()
        logging.info("Setting up FSDP (FSDP2)")

        # Pretrained weights are not DCP format, so they must load before the wrap turns
        # parameters into DTensors.
        self._load_initial_model_weights()

        # FSDP2 applies fully_shard to the model in place.
        self._model_to_fsdp()

        if self.rank == 0:
            logging.info("Model has been wrapped with FSDP")

        # Optimizers must be constructed after wrapping, so they see the sharded parameters.
        self._construct_optimizers()

    def _load_initial_model_weights(self):
        """Initialise DINOv3 and/or model weights when configured.

        Loading is not FSDP-aware, so it has to happen before the model is sharded.
        Full-model weights are loaded last and therefore take precedence over DINOv3 weights
        for overlapping parameters.
        """
        dinov3_path = self.checkpoint_conf.dinov3_weight_path
        if dinov3_path is not None:
            logging.info(f"Initialising aggregator.patch_embed from DINOv3 weights at {dinov3_path}")
            load_dinov3_weights(
                self.model,
                dinov3_path,
                strict=self.checkpoint_conf.dinov3_weight_strict,
            )

        path = self.checkpoint_conf.model_weight_path
        if path is None:
            return
        logging.info(f"Initialising model weights from {path}")
        load_model_weights(
            self.model,
            path,
            strict=self.checkpoint_conf.model_weight_strict,
        )

    def run(self):
        self._log_data_summary()
        if self.eval_only:
            started = time.time()
            metrics = self.run_validation(self.epoch)
            self._report_epoch(self.epoch, val=metrics, val_seconds=time.time() - started, title="validation only")
            return
        self.run_train()

    def _scene_counts(self, loader_dataset) -> Dict[str, tuple]:
        """Per source (scene name before the first underscore): number of scenes and frames."""
        counts: Dict[str, tuple] = {}
        for base in loader_dataset.dataset.base_dataset.datasets:
            root = getattr(base, "UNIFIED_DIR", None)
            names_file = getattr(base, "image_names_filename", "image_names.json")
            for name in getattr(base, "sequence_list", ()):
                with open(Path(root) / name / names_file) as handle:
                    frames = len(json.load(handle))
                scenes, total = counts.get(name.split("_")[0], (0, 0))
                counts[name.split("_")[0]] = (scenes + 1, total + frames)
        return counts

    def _log_data_summary(self) -> None:
        if self.rank != 0:
            return
        try:
            splits = {"train": self._scene_counts(self.train_dataset)}
            if self.val_dataset is not None:
                splits["val"] = self._scene_counts(self.val_dataset)
        except (OSError, ValueError, AttributeError) as error:
            logging.warning(f"Could not summarize the data: {error}")
            return
        roots = [
            str(getattr(base, "UNIFIED_DIR", ""))
            for base in self.train_dataset.dataset.base_dataset.datasets
        ]
        bags = {
            "train": OmegaConf.to_container(self.data_conf.train.get("mixed_scene") or OmegaConf.create({})),
            "val": OmegaConf.to_container(self.validation_conf.get("mixed_scene") or OmegaConf.create({})),
        }
        logging.info(report.format_data_summary(roots, splits, bags))
        if self.wandb_writer is not None:
            self.wandb_writer.update_config(
                {
                    "data_summary": {
                        split: {source: {"scenes": scenes, "frames": frames} for source, (scenes, frames) in sources.items()}
                        for split, sources in splits.items()
                    }
                }
            )

    def _setup_dataloaders(self):
        self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
        self.val_dataset = None
        if self.validation_conf.get("enabled", False):
            val_conf = OmegaConf.create(OmegaConf.to_container(self.data_conf.train, resolve=True))
            val_conf.common_config.training = False
            val_conf.num_workers = int(self.validation_conf.get("num_workers", 2))
            val_conf.seed = int(self.validation_conf.get("seed", 7))
            if self.validation_conf.get("mixed_scene") is not None:
                val_conf.mixed_scene = OmegaConf.create(
                    OmegaConf.to_container(self.validation_conf["mixed_scene"], resolve=True)
                )
            self.val_dataset = instantiate(val_conf, _recursive_=False)

    def _validation_due(self) -> bool:
        if self.val_dataset is None:
            return False
        every = int(self.validation_conf.get("every_n_epochs", 1))
        return (int(self.epoch) + 1) % every == 0 or int(self.epoch) + 1 == self.max_epochs

    def _progress(self, total: int, desc: str) -> Progress:
        """Rank 0 shows progress. A terminal gets a bar, a log file gets a line every `console_every` steps."""
        every = int(getattr(self.logging_conf, "console_every", 100))
        return Progress(total, desc, enabled=self.rank == 0, every=every)

    @staticmethod
    def _running_averages(meters) -> Dict[str, float]:
        shown = {}
        for label, name in (
            ("loss", "train_loss_objective"),
            ("ARI", "train_ari"),
            ("perfect bags", "train_perfect_bags"),
            ("pairs right", "train_pair_acc"),
        ):
            meter = meters.get(name)
            if meter is not None and meter.count:
                shown[label] = meter.avg
        return shown

    @staticmethod
    def _stop_workers(iterator) -> None:
        """Stop the workers of a loader iterator that is left early, before it is garbage collected."""
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is None:
            return
        try:
            shutdown()
        except RuntimeError as error:
            logging.warning(f"A data worker exited abnormally while the loader was closed: {error}")

    @contextlib.contextmanager
    def _worker_scope(self, loader):
        iterator = iter(loader)
        try:
            yield iterator
        finally:
            self._stop_workers(iterator)

    @torch.no_grad()
    def run_validation(self, epoch: int) -> Dict[str, float]:
        """Score mixed bags with the predicted, oracle and plain systems. Returns rank-averaged metrics."""
        if self.val_dataset is None:
            raise RuntimeError("Validation is not configured. Set validation.enabled=true.")
        from multiscene_eval import run_systems

        model = unwrap_ddp_if_wrapped(self.model)
        was_training = model.training
        model.eval()
        set_seeds(int(self.validation_conf.get("seed", 7)), self.max_epochs, self.rank)
        num_batches = int(self.validation_conf.get("num_batches", 50))
        include_separate = bool(self.validation_conf.get("include_separate", False))

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        first_bag = None
        desc = f"validation {int(epoch) + 1}/{self.max_epochs}"
        progress = self._progress(num_batches, desc)
        with self._worker_scope(self.val_dataset.get_loader(epoch=0)) as batches:
            for index, batch in enumerate(batches):
                if index >= num_batches:
                    break
                batch = copy_data_to_device(batch, self.device, non_blocking=True)
                for row in range(batch["images"].shape[0]):
                    with torch.autocast(
                        device_type="cuda",
                        enabled=self.optim_conf.amp.enabled,
                        dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                    ):
                        try:
                            by_system = run_systems(
                                model,
                                batch["images"][row],
                                batch["scene_id"][row],
                                batch["extrinsics"][row],
                                batch["depths"][row],
                                include_separate=include_separate,
                            )
                        except ValueError as error:
                            logging.warning(f"Skipping validation bag: {error}")
                            continue
                    for system, scores in by_system.items():
                        for key, value in scores.items():
                            name = f"{system}/{key}"
                            sums[name] = sums.get(name, 0.0) + float(value)
                            counts[name] = counts.get(name, 0) + 1
                    sums["stats/num_scenes"] = sums.get("stats/num_scenes", 0.0) + float(batch["num_scenes"][row])
                    counts["stats/num_scenes"] = counts.get("stats/num_scenes", 0) + 1
                    if first_bag is None and self.rank == 0:
                        first_bag = (batch["images"][row], batch["scene_id"][row])
                running = {
                    name: sums[f"predicted/{key}"] / counts[f"predicted/{key}"]
                    for name, key in (("ARI", "ari"), ("perfect bags", "perfect_bags"))
                    if counts.get(f"predicted/{key}")
                }
                progress.step(index + 1, running)
        progress.close()

        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (sums, counts))
        total_sums: Dict[str, float] = {}
        total_counts: Dict[str, int] = {}
        for rank_sums, rank_counts in gathered:
            for name, value in rank_sums.items():
                total_sums[name] = total_sums.get(name, 0.0) + value
                total_counts[name] = total_counts.get(name, 0) + rank_counts[name]
        metrics = {name: total_sums[name] / total_counts[name] for name in total_sums}
        metrics["stats/num_bags"] = float(total_counts.get("stats/num_scenes", 0))

        if first_bag is not None and self.wandb_writer is not None and not self._is_fsdp_training():
            self._log_affinity_images(model, *first_bag, epoch)
        model.train(was_training)
        return metrics

    def _log_affinity_images(self, model, images: torch.Tensor, scene_id: torch.Tensor, epoch: int) -> None:
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            prediction = model(images.unsqueeze(0))
        target = (scene_id[:, None] == scene_id[None, :]).float()
        self.wandb_writer.log_affinity(
            {
                "final": prediction["affinity"][0].float().cpu().numpy(),
                "route": prediction["route_affinity"][0].float().cpu().numpy(),
                "target": target.cpu().numpy(),
            },
            epoch,
        )

    def _report_epoch(
        self,
        epoch: int,
        train: Optional[Dict[str, float]] = None,
        val: Optional[Dict[str, float]] = None,
        train_seconds: float = 0.0,
        val_seconds: float = 0.0,
        title: Optional[str] = None,
    ) -> None:
        """Print one block and write one wandb row for the epoch."""
        if self.rank != 0:
            return
        eta = max(0, self.max_epochs - epoch - 1) * (train_seconds + val_seconds)
        logging.info(
            report.format_epoch(epoch, self.max_epochs, train, val, train_seconds, val_seconds, eta, title=title)
        )
        if self.wandb_writer is None:
            return
        payload = report.train_scalars(train) if train else {}
        if val:
            payload.update(report.val_scalars(val))
        self.wandb_writer.log_epoch(payload, epoch, report.val_comparisons(val) if val else None)

    def run_train(self):
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch*100, self.max_epochs, self.rank)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            started = time.time()
            train_metrics = self.train_epoch(dataloader)
            train_seconds = time.time() - started

            dist.barrier()
            self.save_checkpoint(self.epoch)
            val_metrics, val_seconds = None, 0.0
            if self._validation_due():
                started = time.time()
                val_metrics = self.run_validation(int(self.epoch))
                val_seconds = time.time() - started
            self._report_epoch(int(self.epoch), train_metrics, val_metrics, train_seconds, val_seconds)

            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.epoch += 1

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Run one epoch and return the epoch average of every logged scalar, plus the learning rate."""
        # WARNING: These timing values are only useful as averages
        # there aren't `torch.cuda.synchronize()` calls around the timing calls
        # which means the times can undercount work from the current step while also
        # counting work from the previous step
        batch_time = AverageMeter("Batch Time", ":.2f")
        data_time = AverageMeter("Data Time", ":.2f")
        mem = AverageMeter("Mem (GB)", ":.2f")
        phase = "train"

        iters_per_epoch = len(train_loader)

        scalar_names = self._get_scalar_log_keys(phase)
        scalar_names = [f"{phase}_{name}" for name in scalar_names]
        scalar_meters = OrderedDict(
            [(name, AverageMeter(name, ":.4f")) for name in scalar_names]
        )
        grad_meters = OrderedDict()

        if self.gradient_clipper is not None:
            for config in self.gradient_clipper.configs:
                param_names = ",".join(config["module_names"])
                grad_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", ":.2f")

        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )

        assert limit_train_batches <= iters_per_epoch, "limit_train_batches is greater than iters_per_epoch"

        step_stat_meters = [
            batch_time,
            data_time,
            mem,
            self.time_elapsed_meter,
            *grad_meters.values(),
        ]
        self.model.train()

        end = time.time()

        iters_done = 0
        train_bar = self._progress(limit_train_batches, f"train {int(self.epoch) + 1}/{self.max_epochs}")
        batches = iter(train_loader)
        for data_iter, batch in enumerate(batches):
            skip_step_logging = data_iter == 0
            if data_iter >= limit_train_batches:
                break

            # measure data loading time
            data_loading_duration = time.time() - end
            if not skip_step_logging:
                data_time.update(data_loading_duration)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps
            chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            # Update training progress (where) BEFORE the step so loss functions use the correct value
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.schedule_progress = float(exact_epoch) / self.max_epochs
            assert self.schedule_progress <= 1 + self.EPSILON

            scalar_values, scalar_weights = self._run_steps_on_batch_chunks(
                chunked_batches,
                phase,
                log_first_iter_backward=(data_iter == 0),
            )
            for key, value in scalar_values.items():
                scalar_meters[f"{phase}_{key}"].update(value, scalar_weights[key])
            train_bar.step(data_iter + 1, self._running_averages(scalar_meters))

            step = self.steps[phase]
            if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                for key, value in scalar_values.items():
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

            # fvcore's schedulers reject where == 1.0, so skip the final update.
            if self.schedule_progress < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.schedule_progress)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.schedule_progress} of [0,1]."
                )

            # Log schedulers
            if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                step,
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "schedule_progress"),
                    self.schedule_progress,
                    step,
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)
                grad_norm_dict = self.gradient_clipper()
                large_grad_entries = []

                for key, grad_norm in grad_norm_dict.items():
                    if torch.is_tensor(grad_norm):
                        grad_norm = grad_norm.item()
                    grad_meters[f"Grad/{key}"].update(grad_norm)
                    if grad_norm > 500:
                        large_grad_entries.append(f"{key}={grad_norm:.2f}")

                if large_grad_entries:
                    seq_names = batch.get("seq_name", "unknown")
                    if isinstance(seq_names, np.ndarray):
                        seq_names = seq_names.tolist()
                    elif isinstance(seq_names, str):
                        seq_names = [seq_names]
                    elif isinstance(seq_names, tuple):
                        seq_names = list(seq_names)
                    elif not isinstance(seq_names, list):
                        seq_names = [str(seq_names)]
                    logging.warning(
                        f"Large gradients detected on rank {self.rank}: "
                        f"{', '.join(large_grad_entries)}; seq_name={seq_names}"
                    )
            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # measure elapsed time
            iter_duration = time.time() - end

            end = time.time()
            if not skip_step_logging:
                batch_time.update(iter_duration)

            self.time_elapsed_meter.update(
                time.time() - self.start_time
            )

            if not skip_step_logging:
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            # Log progress meters.
            if (
                not skip_step_logging
                and step % self.logging_conf.log_freq == 0
                and self.rank == 0
            ):
                for progress_meter in step_stat_meters:
                    self.tb_writer.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        step,
                    )

            self.steps[phase] += 1
            iters_done = data_iter + 1

        self._check_epoch_length_agreement(iters_done, limit_train_batches)

        train_bar.close()
        self._stop_workers(batches)

        self.est_epoch_time["train"] = batch_time.avg * limit_train_batches
        self._log_timers("train")
        return self._epoch_averages(scalar_meters)

    def _check_epoch_length_agreement(self, iters_done: int, limit_train_batches: int) -> None:
        """Fail loudly when a rank ran fewer iterations than the rest of the world.

        The dataloader reports a nominal length of 1e6, so limit_train_batches cannot be
        validated up front. When the data runs out the ranks leave the epoch at different
        iterations and the next collective has nobody to talk to, which looks like an
        unexplained hang rather than a configuration error.

        The local branch runs first because it needs no collective: a rank that came up
        short reports it even while the others are still waiting inside backward. The
        all_reduce afterwards covers the case where every rank finished but on different
        counts, which the local check cannot see.
        """
        if iters_done < limit_train_batches:
            raise RuntimeError(
                f"Rank {self.rank} ran {iters_done} of {limit_train_batches} "
                f"limit_train_batches in epoch {self.epoch}: the dataloader ran out of data. "
                f"The dataset is too small for this world size -- raise samples_per_weight_unit or "
                f"mixture_weight_scale, add datasets, or lower limit_train_batches."
            )

        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        counts = torch.tensor([iters_done, -iters_done], device=self.device)
        dist.all_reduce(counts, op=dist.ReduceOp.MAX)
        highest, lowest = int(counts[0]), -int(counts[1])
        if highest != lowest:
            raise RuntimeError(
                f"Ranks disagree on epoch {self.epoch} length: between {lowest} and {highest} "
                f"iterations were run. Every rank must run the same number of steps or the "
                f"next collective will deadlock."
            )

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        log_first_iter_backward: bool = False,
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """
        for optim in self.optims:
            optim.optimizer.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)
        scalar_sums = {}
        scalar_weights = {}

        for i, chunked_batch in enumerate(chunked_batches):

            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )


            with ddp_context:
                loss_dict, scalar_values, batch_size = self._step(
                    chunked_batch,
                    self.model,
                    phase,
                )

                for key, value in scalar_values.items():
                    scalar_sums[key] = scalar_sums.get(key, 0.0) + value * batch_size
                    scalar_weights[key] = scalar_weights.get(key, 0) + batch_size


                loss = loss_dict["loss_objective"]

                loss = loss / accum_steps

                if log_first_iter_backward:
                    logging.info(
                        f"[Rank {self.rank}] About to run backward on first iter "
                        f"(chunk {i + 1}/{accum_steps})"
                    )
                self.scaler.scale(loss).backward()
                if log_first_iter_backward:
                    logging.info(
                        f"[Rank {self.rank}] Finished backward on first iter "
                        f"(chunk {i + 1}/{accum_steps})"
                    )

        scalar_values = {
            key: value / scalar_weights[key] for key, value in scalar_sums.items()
        }
        return scalar_values, scalar_weights


    def _log_timers(self, phase):
        if self.rank == 0:
            self.tb_writer.log(
                os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
                self.time_elapsed_meter.val,
                self.steps[phase],
            )

    def _epoch_averages(self, scalar_meters) -> Dict[str, float]:
        """Epoch average of each logged scalar, keyed by its plain name, plus the mean learning rate."""
        averages = {
            name[len("train_") :]: meter.avg
            for name, meter in scalar_meters.items()
            if name.startswith("train_") and meter.count
        }
        learning_rates = [group["lr"] for optim in self.optims for group in optim.optimizer.param_groups]
        if learning_rates:
            averages["lr"] = sum(learning_rates) / len(learning_rates)
        return averages

    def _setup_components(self):
        logging.info("Setting up components: Model, loss, optim, meters etc.")

        self.epoch = self.start_epoch
        self.steps = {"train": 0}

        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer)
        self.wandb_writer = (
            instantiate(self.logging_conf.wandb_writer)
            if self.logging_conf.wandb_writer is not None
            else None
        )
        if self.wandb_writer is not None:
            self.wandb_writer.update_config(
                {
                    name: OmegaConf.to_container(section, resolve=True)
                    for name, section in (
                        ("model", self.model_conf),
                        ("loss", self.loss_conf),
                        ("optim", self.optim_conf_raw),
                        ("data", self.data_conf),
                    )
                    if section is not None
                }
            )
        self.model = instantiate(self.model_conf, _recursive_=False)


        if self.optim_conf.frozen_module_names is not None:
            logging.info(
                f"[Start] Freezing 'frozen_module_names' on rank {self.rank}"
            )
            self.model = freeze_modules(
                self.model, name_patterns=self.optim_conf.frozen_module_names
            )
            logging.info(
                f"[Done] Freezing 'frozen_module_names' on rank {self.rank}"
            )
        if self.optim_conf.frozen_param_names is not None:
            logging.info(
                f"[Start] Freezing 'frozen_param_names' on rank {self.rank}"
            )
            self.model = freeze_parameters(
                self.model, name_patterns=self.optim_conf.frozen_param_names
            )
            logging.info(
                f"[Done] Freezing 'frozen_param_names' on rank {self.rank}"
            )

        self.loss = None
        if self.loss_conf:
            self.loss = instantiate(self.loss_conf, _recursive_=False)


        # FDSP needs a different Gradient Scaler than DDP
        if self._is_fsdp_training():
            self.scaler = create_grad_scaler(
                self.fsdp_settings, enabled=self.optim_conf.amp.enabled
            )
        else:
            if "bfloat16" in self.optim_conf.amp.amp_dtype:
                self.scaler = torch.amp.GradScaler("cuda", enabled=False)
            else:
                self.scaler = torch.amp.GradScaler("cuda", enabled=self.optim_conf.amp.enabled)

        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _construct_optimizers(self):
        self.optims = construct_optimizers(self.model, self.optim_conf)

    def _step(
        self,
        batch: Mapping,
        model: nn.Module,
        phase: str,
    ):

        model_inner = unwrap_ddp_if_wrapped(model)
        if hasattr(model_inner, "set_group_progress"):
            model_inner.set_group_progress(self.schedule_progress)

        with torch.autocast(
            device_type="cuda",
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            scene_id = batch.get("scene_id")
            if scene_id is None:
                y_hat = model(batch["images"])
            else:
                y_hat = model(batch["images"], scene_id=scene_id)

        with torch.autocast(device_type="cuda", enabled=False):
            # The camera loss weights a sequence of pose predictions, one per refinement stage.
            # This head emits a single stage, so wrap it into the list the loss expects.
            if "pose_enc" in y_hat:
                y_hat["pose_enc_list"] = [y_hat["pose_enc"]]
            if hasattr(model_inner, "gt_group_prob"):
                y_hat["gt_group_prob"] = torch.tensor(model_inner.gt_group_prob)
            loss_dict = self.loss(y_hat, batch, self.schedule_progress)

        log_data = {**y_hat, **loss_dict, **batch}
        scalar_values, batch_size = self._collect_scalar_values(log_data, phase)

        return loss_dict, scalar_values, batch_size


    def _collect_scalar_values(
        self,
        batch: Mapping,
        phase: str,
    ) -> tuple[dict[str, float], int]:
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = batch["extrinsics"].shape[0]
        scalar_values = {}
        for key in keys_to_log:
            if key in batch:
                value = batch[key].item() if torch.is_tensor(batch[key]) else batch[key]
                scalar_values[key] = float(value)
        return scalar_values, batch_size
