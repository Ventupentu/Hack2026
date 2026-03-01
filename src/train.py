"""Hydra entrypoint for retrieval training."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from src.config import InditexConfig
from src.models import train_retrieval_model


cs = ConfigStore.instance()
cs.store(name="inditex_config", node=InditexConfig)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: InditexConfig) -> None:
    """Resolve paths from Hydra and launch selected trainer backend."""
    files = cfg.files

    data_dir = Path(to_absolute_path(files.data_dir))
    train_manifest = data_dir / "bundles_product_match_train.csv"
    val_manifest = data_dir / "bundles_product_match_train.csv"
    products_manifest = data_dir / "product_dataset_with_gender.csv"
    bundles_images_dir = Path(to_absolute_path(files.bundles_images))
    products_images_dir = Path(to_absolute_path(files.products_images))
    yolo_detections_dir = Path(to_absolute_path(files.yolo_detections_dir))
    model_name = str(getattr(cfg.params, "model_name", "openclip_marqo_siglip")).strip()
    if model_name == "openclip_marqo_siglip":
        model_slug = "retrieval_openclip"
    else:
        model_slug = f"retrieval_{model_name.replace('-', '_')}"

    # Per-run output dir (logs, metrics for this run)
    output_dir = Path(HydraConfig.get().runtime.output_dir) / model_slug
    # Persistent checkpoint dir (shared across runs, enables resume)
    checkpoint_dir = Path(to_absolute_path(f"outputs/{model_slug}"))

    train_retrieval_model(
        cfg=cfg,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        products_manifest=products_manifest,
        bundles_images_dir=bundles_images_dir,
        products_images_dir=products_images_dir,
        output_dir=output_dir,
        cache_dir=yolo_detections_dir,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
