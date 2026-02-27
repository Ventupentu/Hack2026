"""Pretrained feature extractor builders."""

from __future__ import annotations

from typing import Callable, Tuple

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


def build_pretrained_encoder(model_name: str, device: torch.device) -> Tuple[nn.Module, Callable, int]:
    """Build a torchvision encoder and matching preprocessing pipeline."""
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
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Use one of: resnet18, resnet50, efficientnet_b0."
        )

    model.eval().to(device)
    preprocess = weights.transforms(antialias=True)
    return model, preprocess, embedding_dim
