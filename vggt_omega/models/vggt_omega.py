# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead
from vggt_omega.models.multiscene import MultiScene, choose_groups, predict_pose_by_scene
from vggt_omega.utils.scenes import pack_scenes


class VGGTOmega(nn.Module):
    """VGGT-Omega camera and depth model, optionally with multi-scene routing."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        use_checkpoint: bool = False,
        autocast: bool = True,
        multiscene: dict | None = None,
    ) -> None:
        super().__init__()

        self.autocast = autocast
        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim, use_checkpoint=use_checkpoint)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        self.multiscene = _build_multiscene(multiscene, dim=2 * embed_dim)

        self.isolate_backbone = True
        self.isolate_camera = True
        self.gt_group_prob_start = 0.0
        self.gt_group_prob_end = 0.0
        self.gt_group_prob = 0.0
        if self.multiscene is not None:
            self.isolate_backbone = bool(multiscene.get("isolate_backbone", True))
            self.isolate_camera = bool(multiscene.get("isolate_camera", True))
            start, end = multiscene.get("gt_group_prob", [0.0, 0.0])
            self.gt_group_prob_start, self.gt_group_prob_end = float(start), float(end)
            self.set_group_progress(0.0)

    def set_group_progress(self, progress: float) -> None:
        """Anneal the chance of using ground-truth groups while training. `progress` is in [0, 1]."""
        progress = min(max(float(progress), 0.0), 1.0)
        self.gt_group_prob = self.gt_group_prob_start + progress * (self.gt_group_prob_end - self.gt_group_prob_start)

    def forward(
        self,
        images: torch.Tensor,
        scene_id: torch.Tensor | None = None,
        use_gt_groups: bool = False,
        isolate: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Predict camera and depth for a bag of frames that may come from several scenes.

        `scene_id` (B, N) holds ground-truth labels. Training uses them as the groups with
        probability `gt_group_prob`, and `use_gt_groups=True` forces them. Without labels the
        predicted groups are used. `isolate=False` keeps every frame in one group.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        scene_model = self.multiscene
        use_gt = (
            scene_model is not None
            and isolate
            and scene_id is not None
            and (use_gt_groups or (self.training and bool(torch.rand(()) < self.gt_group_prob)))
        )

        route_state = {}
        router = None
        route_layer = None
        if scene_model is not None:
            route_layer = scene_model.route_layer

            def router(scene_tokens):
                route_state.update(scene_model.route(scene_tokens))
                if not (isolate and self.isolate_backbone):
                    return None
                return choose_groups(route_state["route_group_id"], scene_id, use_gt)

        if self.autocast:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            aggregator_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            aggregator_context = contextlib.nullcontext()
        with aggregator_context:
            aggregated_tokens_list, patch_token_start = self.aggregator(
                images, router=router, route_layer=route_layer
            )

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {}
        camera_groups = None
        if scene_model is not None:
            final_tokens, scene_predictions = scene_model(
                final_tokens,
                patch_token_start=patch_token_start,
                scene_id=scene_id,
                use_gt=use_gt,
                isolate=isolate,
            )
            aggregated_tokens_list[-1] = final_tokens
            predictions.update(route_state)
            predictions.update(scene_predictions)
            predictions["used_gt_groups"] = torch.tensor(float(use_gt), device=images.device)
            camera_groups = scene_predictions["used_group_id"]

        predictions["camera_and_register_tokens"] = final_tokens[:, :, :patch_token_start].contiguous()
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                if camera_groups is not None and self.isolate_camera:
                    predictions["pose_enc"] = predict_pose_by_scene(
                        self.camera_head,
                        final_tokens,
                        patch_token_start,
                        camera_groups,
                    )
                else:
                    predictions["pose_enc"] = self.camera_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions

    @torch.inference_mode()
    def predict_scenes(self, images: torch.Tensor) -> list[dict]:
        """Group a bag of frames into scenes and return one entry per scene.

        `images` is (N, 3, H, W) or (1, N, 3, H, W). Each entry has `images` (frame indices),
        `anchor`, `camera_poses` relative to the anchor, and `pointcloud` in the anchor frame.
        """
        if self.multiscene is None:
            raise RuntimeError("predict_scenes needs the model to be built with multiscene enabled")
        was_training = self.training
        self.eval()
        try:
            if images.ndim == 4:
                images = images.unsqueeze(0)
            if images.shape[0] != 1:
                raise ValueError(f"predict_scenes takes one bag at a time, got batch size {images.shape[0]}")
            prediction = self(images)
        finally:
            self.train(was_training)
        return pack_scenes(
            prediction["affinity"][0],
            prediction["used_group_id"][0],
            prediction["pose_enc"][0],
            prediction["depth"][0],
            tuple(images.shape[-2:]),
        )


def _build_multiscene(config: dict | None, dim: int) -> MultiScene | None:
    """Build the multi-scene block only when the config asks for it.

    Leaving `multiscene` unset keeps the released checkpoint path unchanged.
    """
    if config is None or not bool(config.get("enabled", False)):
        return None
    return MultiScene(
        dim=dim,
        num_heads=int(config.get("num_heads", 16)),
        threshold=float(config.get("threshold", 0.5)),
        linkage=str(config.get("linkage", "average")),
        route_layer=int(config.get("route_layer", 11)),
    )


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
