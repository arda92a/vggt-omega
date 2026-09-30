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
from vggt_omega.models.multiscene import MultiScene, predict_pose_by_scene, select_camera_groups


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

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
        self.isolate_camera = False
        self.camera_groups = "gt"
        if self.multiscene is not None:
            self.isolate_camera = bool(multiscene.get("isolate_camera", True))
            self.camera_groups = str(multiscene.get("camera_groups", "gt"))
            if self.camera_groups not in ("gt", "predicted"):
                raise ValueError(
                    f"model.multiscene.camera_groups must be 'gt' or 'predicted', got {self.camera_groups!r}"
                )

    def forward(self, images: torch.Tensor, scene_id: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if self.autocast:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            aggregator_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            aggregator_context = contextlib.nullcontext()
        with aggregator_context:
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {}
        if self.multiscene is not None:
            final_tokens, scene_predictions = self.multiscene(
                final_tokens,
                patch_token_start=patch_token_start,
                hard_mask=not self.training,
            )
            aggregated_tokens_list[-1] = final_tokens
            predictions.update(scene_predictions)

        predictions["camera_and_register_tokens"] = final_tokens[:, :, :patch_token_start].contiguous()
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                if self.isolate_camera:
                    group_id = select_camera_groups(
                        predictions["group_id"],
                        scene_id,
                        self.camera_groups,
                    )
                    predictions["pose_enc"] = predict_pose_by_scene(
                        self.camera_head,
                        final_tokens,
                        patch_token_start,
                        group_id,
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
        bias_scale=float(config.get("bias_scale", 1.0)),
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
