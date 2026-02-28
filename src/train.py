"""Hydra entrypoint for retrieval training."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from config import InditexConfig
from models import train_retrieval_model


cs = ConfigStore.instance()
cs.store(name="inditex_config", node=InditexConfig)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: InditexConfig) -> None:
    """Resolve paths from Hydra and launch selected trainer backend."""
    files = cfg.files

    train_manifest = Path(to_absolute_path(files.train_manifest))
    val_manifest = Path(to_absolute_path(files.val_manifest))
    products_manifest = Path(to_absolute_path(files.products_manifest))
    bundles_images_dir = Path(to_absolute_path(files.bundles_images))
    products_images_dir = Path(to_absolute_path(files.products_images))
    output_dir = Path(to_absolute_path(files.output_dir))

    train_retrieval_model(
        cfg=cfg,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        products_manifest=products_manifest,
        bundles_images_dir=bundles_images_dir,
        products_images_dir=products_images_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
