"""Scene router: DINOv3 features per frame, a small transformer over the frames of a bag, a pairwise affinity matrix.

    frame -> DINOv3 (CLS + mean patch) -> MLP -> bag transformer (frames see each other) -> embedding e_i
    affinity logit(i, j) = scale * cos(e_i, e_j) + bias        sigmoid of it: P(i and j are the same sequence)

The backbone comes from the Hugging Face Hub, for example facebook/dinov3-vits16-pretrain-lvd1689m (any DINOv3 ViT
works, also vitb16, vitl16, vits16plus, vith16plus). These repos are gated: request access on the model page once,
then log in on the machine with `hf auth login` or set HF_TOKEN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_backbone(name: str) -> nn.Module:
    # DINOv3 jitters patch positions in train mode (pos_embed_rescale=2), an augmentation of its own pretraining.
    # It is turned off so fine-tuning sees the same positions as inference.
    return AutoModel.from_pretrained(name, pos_embed_rescale=None)


def set_trainable(backbone: nn.Module, train_blocks: int) -> None:
    """Freeze the backbone except its last `train_blocks` blocks and final norm (0 freezes all of it)."""
    backbone.requires_grad_(False)
    if train_blocks > 0:
        for block in backbone.model.layer[-train_blocks:]:
            block.requires_grad_(True)
        backbone.norm.requires_grad_(True)


class SceneRouter(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int = 256, context_layers: int = 2, heads: int = 8):
        super().__init__()
        self.backbone = backbone
        self.first_patch = 1 + backbone.config.num_register_tokens  # tokens are [CLS, registers, patches]
        width = backbone.config.hidden_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.project = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, dim), nn.GELU(), nn.Linear(dim, dim))
        self.context = None
        if context_layers > 0:
            layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, context_layers, enable_nested_tensor=False)
        self.embed = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def frame_features(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 images (M, 3, H, W) -> (M, 2 * width): CLS token and mean patch token."""
        x = (images.float() / 255 - self.mean) / self.std
        tokens = self.backbone(pixel_values=x).last_hidden_state  # already layer-normed
        return torch.cat([tokens[:, 0], tokens[:, self.first_patch:].mean(1)], dim=-1).float()

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """images (B, N, 3, H, W) with mask (B, N) of real frames -> affinity logits (B, N, N)."""
        batch, frames = mask.shape
        features = self.project(self.frame_features(images[mask]))  # padded frames never reach the backbone
        tokens = features.new_zeros(batch, frames, features.shape[-1])
        tokens[mask] = features
        if self.context is not None:
            tokens = self.context(tokens, src_key_padding_mask=~mask)
        embeddings = F.normalize(self.embed(tokens), dim=-1)
        return self.scale * embeddings @ embeddings.transpose(1, 2) + self.bias


def build_router(cfg: dict) -> SceneRouter:
    backbone = load_backbone(cfg["backbone"])
    set_trainable(backbone, cfg["train_blocks"])
    return SceneRouter(backbone, dim=cfg["dim"], context_layers=cfg["context_layers"])
