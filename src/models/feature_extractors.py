"""Pretrained feature extractor builders."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torchvision import models


def resolve_device(device: str) -> torch.device:
    """Resolve runtime device from CLI value."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no CUDA device is available.")
    return torch.device(device)


class _HFClipImageEncoder(nn.Module):
    """Adapter to expose image features as a plain tensor forward pass."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.hf_model.get_image_features(pixel_values=pixel_values)


def _build_hf_clip_encoder(model_id: str, device: torch.device) -> Tuple[nn.Module, Callable, int]:
    """Build a HuggingFace CLIP-like image encoder."""
    try:
        from transformers import AutoModel, AutoProcessor
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise ModuleNotFoundError(
            "Model requires transformers. Install with:\n"
            "  pip install transformers"
        ) from exc

    hf_model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = _HFClipImageEncoder(hf_model)
    model.eval().to(device)

    def preprocess(image):
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        return pixel_values[0]

    projection_dim = getattr(hf_model.config, "projection_dim", None)
    hidden_size = getattr(hf_model.config, "hidden_size", None)
    embedding_dim = int(projection_dim or hidden_size or 512)
    return model, preprocess, embedding_dim


def build_pretrained_encoder(
    model_name: str,
    device: torch.device,
    hf_model_id: Optional[str] = None,
) -> Tuple[nn.Module, Callable, int]:
    """Build an encoder and matching preprocessing pipeline."""
    name = model_name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        embedding_dim = 512
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        embedding_dim = 2048
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        embedding_dim = 1280
    elif name in {"fashionclip", "hf_clip", "clip"}:
        if hf_model_id and hf_model_id.strip():
            model_id = hf_model_id.strip()
        elif name == "fashionclip":
            model_id = "patrickjohncyh/fashion-clip"
        else:
            model_id = "openai/clip-vit-base-patch32"
        return _build_hf_clip_encoder(model_id=model_id, device=device)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Use one of: resnet18, resnet50, efficientnet_b0, fashionclip, clip."
        )

    model.eval().to(device)
    preprocess = weights.transforms(antialias=True)
    return model, preprocess, embedding_dim
