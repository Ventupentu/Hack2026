from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

LOGGER = logging.getLogger(__name__)


def _resolve_image_size(model: torch.nn.Module) -> int:
    image_size = getattr(getattr(model, "visual", None), "image_size", 224)
    if isinstance(image_size, tuple):
        return int(image_size[0])
    return int(image_size)


def build_query_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ]
    )


def build_product_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )


class FashionSigLIPEncoder:
    """OpenCLIP wrapper for Marqo fashionSigLIP image embeddings."""

    def __init__(
        self,
        model_name: str = "hf-hub:Marqo/marqo-fashionSigLIP",
        device: str | None = None,
        trainable: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        LOGGER.info("Loading OpenCLIP model %s on %s", model_name, self.device)
        model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
        model = model.to(self.device)
        model.train(mode=trainable)

        self.model = model
        self.preprocess_val = preprocess_val
        self.image_size = _resolve_image_size(model)
        self.query_train_transform = build_query_train_transform(self.image_size)
        self.product_train_transform = build_product_train_transform(self.image_size)

    @property
    def embedding_dim(self) -> int:
        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            return int(self.model.text_projection.shape[-1])
        dummy = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)
        with torch.no_grad():
            emb = self.encode_tensor(dummy)
        return int(emb.shape[-1])

    def set_trainable(self, trainable: bool) -> None:
        self.model.train(mode=trainable)
        for param in self.model.parameters():
            param.requires_grad = trainable

    def encode_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode tensor batch [B,3,H,W] and L2 normalize."""
        feats = self.model.encode_image(image_tensor)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_pil_batch(
        self,
        images: Iterable[Image.Image],
        transform: transforms.Compose | None = None,
        batch_size: int = 64,
    ) -> torch.Tensor:
        used_transform = transform or self.preprocess_val
        tensors = [used_transform(img) for img in images]
        if not tensors:
            return torch.empty((0, self.embedding_dim), device=self.device)

        chunks: list[torch.Tensor] = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size], dim=0).to(self.device)
            feats = self.encode_tensor(batch)
            chunks.append(feats)
        return torch.cat(chunks, dim=0)

    def save_checkpoint(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "extra": extra or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict:
        payload = torch.load(path, map_location=self.device)
        state_dict = payload.get("state_dict", payload)
        self.model.load_state_dict(state_dict, strict=strict)
        LOGGER.info("Loaded encoder checkpoint from %s", path)
        return payload.get("extra", {})
