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
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf


from train_utils.checkpoint import (
    DDPCheckpointSaver,
    load_checkpoint as load_checkpoint_file,
    load_dinov3_weights,
    load_model_weights,
    move_optimizer_state_to_device,
    restore_optimizer_state,
)
from train_utils.distributed import (
    get_machine_local_and_dist_rank,
    setup_distributed_backend,
    unwrap_ddp_if_wrapped,
)
from train_utils.freeze import freeze_modules, freeze_parameters
from train_utils.fsdp import (
    ShardedCheckpointLoader,
    ShardedCheckpointSaver,
    apply_fsdp2_strategy,
    create_grad_scaler,
    to_fsdp_settings,
)
from train_utils.general import (
    AverageMeter,
    DurationMeter,
    ProgressMeter,
    copy_data_to_device,
    get_amp_type,
    get_resume_checkpoint,
    human_readable_time,
    makedir,
    set_seeds,
)
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
            self.load_checkpoint()
            self._setup_ddp_distributed_training(distributed)

        if self.gradient_clipper is not None:
            self.gradient_clipper.setup_clipping(self.model)
        dist.barrier()

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
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
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

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
            # For DDP, we save the optimizer state dicts directly
            optimizer_state = [optim.optimizer.state_dict() for optim in self.optims]
            if len(self.optims) == 1:
                optimizer_state = optimizer_state[0]
            checkpoint_content["optimizer"] = optimizer_state

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

        # Check if we are resuming from a checkpoint
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)

        # 1. Load Pretrained Weights BEFORE wrapping (if NOT resuming)
        # CRITICAL: Pretrained weights (non-DCP format) must be loaded BEFORE FSDP wrap
        # because after wrap, parameters become DTensors and standard load_state_dict fails.
        if ckpt_path is None:
            logging.info("No resume checkpoint found. Loading pretrained weights (if any) BEFORE FSDP wrap.")
            self._load_initial_model_weights()

        # 2. Wrap Model with FSDP2
        # FSDP2 applies fully_shard to the model (in-place)
        self._model_to_fsdp()

        if self.rank == 0:
            logging.info("Model has been wrapped with FSDP")

        # Optimizers must be constructed after wrapping, so they see the sharded parameters.
        self._construct_optimizers()

        # 4. Load Resume Checkpoint (DCP format, after wrap and optimizer init)
        if ckpt_path is not None:
            logging.info(f"Resuming from checkpoint: {ckpt_path}")
            self._load_resuming_checkpoint(ckpt_path)

    def _load_initial_model_weights(self):
        """Initialise DINOv3 and/or model weights when configured.

        Only runs when there is nothing to resume from. Loading is not FSDP-aware, so it has
        to happen before the model is sharded. Full-model weights are loaded last and therefore
        take precedence over DINOv3 weights for overlapping parameters.
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

    def load_checkpoint(self):
        dist.barrier()
        assert (
            not self._is_fsdp_training()
        ), "load_checkpoint is included in the FSDP setup"
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._load_initial_model_weights()
        else:
            self._load_resuming_checkpoint(ckpt_path)
        dist.barrier()


    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")
        # Clear CUDA cache before loading
        torch.cuda.empty_cache()

        checkpoint = {}

        if self._is_fsdp_training():
            loader = ShardedCheckpointLoader()

            checkpoint = loader.load_checkpoint(
                model=self.model,
                optimizers=[optim.optimizer for optim in self.optims],
                checkpoint_path=ckpt_path,
            )
            logging.info(f"Done loading FSDP2 state (rank {self.rank})")

        else:
            checkpoint = load_checkpoint_file(ckpt_path)
            self.model.load_state_dict(checkpoint["model"])

            # Clear model state from checkpoint to free CPU memory
            if "model" in checkpoint:
                del checkpoint["model"]
            gc.collect()
            torch.cuda.empty_cache()

            logging.info(f"Loading the optimizer state dict (rank {self.rank})")
            restore_optimizer_state([optim.optimizer for optim in self.optims], checkpoint["optimizer"])
            for optim in self.optims:
                move_optimizer_state_to_device(optim.optimizer, self.device)

            # Clear optimizer state from checkpoint
            if "optimizer" in checkpoint:
                del checkpoint["optimizer"]

        # Common Metadata Restore
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        elif "prev_epoch" in checkpoint:
            self.epoch = checkpoint["prev_epoch"] + 1

        if "steps" in checkpoint:
            self.steps = checkpoint["steps"]

        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # global GC & CUDA allocator flush
        gc.collect()
        del checkpoint
        torch.cuda.empty_cache()

    def run(self):
        if self.epoch > 0:
            logging.info(f"Resuming training from epoch {self.epoch}")
        if self.eval_only:
            self._log_validation(self.run_validation(self.epoch), self.epoch)
            return
        self.run_train()

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

    def _progress(self, iterable, total, desc):
        """Rank 0 progress bar. Other ranks iterate the loader unchanged."""
        if self.rank != 0:
            return iterable
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True, mininterval=0.5)

    @staticmethod
    def _meter_postfix(meters) -> Dict[str, str]:
        shown = {}
        for label, name in (
            ("loss", "train_loss_objective"),
            ("f1", "train_affinity_f1"),
            ("route", "train_route_f1"),
            ("ari", "train_ari"),
            ("exact", "train_exact_partition"),
        ):
            meter = meters.get(name)
            if meter is not None and meter.count:
                shown[label] = f"{meter.avg:.3f}"
        return shown

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
        loader = self.val_dataset.get_loader(epoch=0)
        desc = "val before training" if epoch < 0 else f"val {int(epoch) + 1}/{self.max_epochs}"
        progress = self._progress(loader, num_batches, desc)
        for index, batch in enumerate(progress):
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
                        report = run_systems(
                            model,
                            batch["images"][row],
                            batch["scene_id"][row],
                            batch["extrinsics"][row],
                            batch["depths"][row],
                            threshold=model.multiscene.threshold,
                            include_separate=include_separate,
                        )
                    except ValueError as error:
                        logging.warning(f"Skipping validation bag: {error}")
                        continue
                for system, scores in report.items():
                    for key, value in scores.items():
                        name = f"{system}/{key}"
                        sums[name] = sums.get(name, 0.0) + float(value)
                        counts[name] = counts.get(name, 0) + 1
                sums["stats/num_scenes"] = sums.get("stats/num_scenes", 0.0) + float(batch["num_scenes"][row])
                counts["stats/num_scenes"] = counts.get("stats/num_scenes", 0) + 1
                if first_bag is None and self.rank == 0:
                    first_bag = (batch["images"][row], batch["scene_id"][row])
            ari_count = counts.get("predicted/ari", 0)
            if ari_count and hasattr(progress, "set_postfix"):
                progress.set_postfix(ari=f"{sums['predicted/ari'] / ari_count:.3f}", refresh=False)
        if hasattr(progress, "close"):
            progress.close()
        del loader

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

    def _log_validation(self, metrics: Dict[str, float], epoch: int) -> None:
        if self.rank != 0:
            return
        payload = {f"val/{name}": value for name, value in metrics.items()}
        summary = ", ".join(
            f"{name}={value:.4f}"
            for name, value in payload.items()
            if name.startswith(("val/predicted/", "val/oracle/"))
            and name.rsplit("/", 1)[1] in ("ari", "auc_30", "abs_rel", "affinity_f1")
        )
        logging.info(f"Epoch [{epoch}] validation: {summary}")
        if self.wandb_writer is not None:
            self.wandb_writer.log_epoch(payload, epoch)

    def run_train(self):
        if (
            self.val_dataset is not None
            and self.validation_conf.get("at_start", True)
            and self.epoch == self.start_epoch
        ):
            self._log_validation(self.run_validation(self.start_epoch - 1), self.start_epoch - 1)
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch*100, self.max_epochs, self.rank)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            self.train_epoch(dataloader)

            dist.barrier()
            self.save_checkpoint(self.epoch)
            if self._validation_due():
                self._log_validation(self.run_validation(int(self.epoch)), int(self.epoch))

            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.epoch += 1

    def train_epoch(self, train_loader):
        logging.info(f"Train Epoch Started: [{self.epoch}]")
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
        progress = ProgressMeter(
            num_batches=limit_train_batches,
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *scalar_meters.values(),
                *grad_meters.values(),
            ],
            prefix="Train | Epoch: [{}]".format(self.epoch),
        )

        self.model.train()

        end = time.time()

        iters_done = 0
        logging.info(f"Starting training at epoch {self.epoch}")
        train_bar = self._progress(
            train_loader,
            limit_train_batches,
            f"train {int(self.epoch) + 1}/{self.max_epochs}",
        )
        for data_iter, batch in enumerate(train_bar):
            skip_step_logging = data_iter == 0
            if data_iter >= limit_train_batches:
                break

            if data_iter == 0 and self.rank == 0:
                logging.info(f"First batch: sequences={batch['seq_name']}, datasets={batch['dataset']}")

            # measure data loading time
            data_loading_duration = time.time() - end
            if not skip_step_logging:
                data_time.update(data_loading_duration)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps
            chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            # Update training progress (where) BEFORE the step so loss functions use the correct value
            # This is crucial for resuming, otherwise the first batch uses where=0.0
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
            if hasattr(train_bar, "set_postfix"):
                train_bar.set_postfix(self._meter_postfix(scalar_meters), refresh=False)

            step = self.steps[phase]
            if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                for key, value in scalar_values.items():
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)
                if self.wandb_writer is not None:
                    step_metrics = {f"train_step/{key}": value for key, value in scalar_values.items()}
                    step_metrics["train_step/lr"] = self.optims[0].optimizer.param_groups[0]["lr"]
                    self.wandb_writer.log_step(step_metrics, step)

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
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if not skip_step_logging:
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if (
                not skip_step_logging
                and data_iter % self.logging_conf.log_freq == 0
                and self.rank == 0
                and not hasattr(train_bar, "set_postfix")
            ):
                progress.display(data_iter)

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

        if hasattr(train_bar, "close"):
            train_bar.close()

        self.est_epoch_time["train"] = batch_time.avg * limit_train_batches
        self._log_timers("train")
        self._log_epoch(scalar_meters)

        logging.info(f"Train Epoch Finished: [{self.epoch}]")

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
        epochs_remaining = self.max_epochs - self.epoch - 1
        time_remaining = epochs_remaining * self.est_epoch_time["train"]

        if self.rank == 0:
            self.tb_writer.log(
                os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
                self.time_elapsed_meter.val,
                self.steps[phase],
            )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _log_epoch(self, scalar_meters):
        """Write one point per finished epoch. The x-axis on wandb is the epoch index."""
        if self.rank != 0:
            return
        metrics = {}
        for name, meter in scalar_meters.items():
            key = name[len("train_") :] if name.startswith("train_") else name
            metrics[f"train/{key}"] = meter.avg
        learning_rates = [
            group["lr"]
            for optim in self.optims
            for group in optim.optimizer.param_groups
        ]
        if learning_rates:
            metrics["train/lr"] = sum(learning_rates) / len(learning_rates)
        summary = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        logging.info(f"Epoch [{self.epoch}] averages: {summary}")
        if self.wandb_writer is not None:
            self.wandb_writer.log_epoch(metrics, int(self.epoch))

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
