"""Model/trainer registry for retrieval pipelines."""

from __future__ import annotations

from pathlib import Path

from src.config import InditexConfig
from src.models.retrieval_grlite import train_grlite_retrieval
from src.models.retrieval_openclip import train_openclip_retrieval


def train_retrieval_model(
    cfg: InditexConfig,
    train_manifest: Path,
    val_manifest: Path,
    products_manifest: Path,
    bundles_images_dir: Path,
    products_images_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    checkpoint_dir: Path | None = None,
) -> None:
    """Dispatch retrieval training by configured model backend."""
    if checkpoint_dir is None:
        checkpoint_dir = output_dir
    model_name = getattr(cfg.params, "model_name", "openclip_marqo_siglip")
    if model_name == "openclip_marqo_siglip":
        train_openclip_retrieval(
            cfg=cfg,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            products_manifest=products_manifest,
            bundles_images_dir=bundles_images_dir,
            products_images_dir=products_images_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
        )
        return

    if model_name in {"gr-lite", "gr_lite", "grlite"}:
        train_grlite_retrieval(
            cfg=cfg,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            products_manifest=products_manifest,
            bundles_images_dir=bundles_images_dir,
            products_images_dir=products_images_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
        )
        return

    raise ValueError(
        f"Unsupported params.model_name='{model_name}'. "
        "Available: openclip_marqo_siglip, gr-lite"
    )
