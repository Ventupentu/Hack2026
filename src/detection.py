from __future__ import annotations

import torch
from pathlib import Path
from typing import List

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from ultralyticsplus import YOLO, render_result

from config import InditexConfig

cs = ConfigStore.instance()
cs.store(name="inditex_config", node=InditexConfig)


def load_bundle_image_ids(bundles_csv: Path, max_images: int | None = None) -> List[str]:
    """
    Lee bundles_dataset.csv y devuelve bundle_asset_id para localizar imagenes locales.
    """
    df = pd.read_csv(bundles_csv)
    ids = df["bundle_asset_id"].astype(str).tolist()
    if max_images is not None:
        ids = ids[:max_images]
    return ids


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: InditexConfig) -> None:
    bundles_csv = Path(to_absolute_path(cfg.files.bundles_dataset))
    bundles_images_dir = Path(to_absolute_path(cfg.files.bundles_images))

    # Detection-specific params
    out_dir = Path(to_absolute_path(cfg.files.yolo_detections_dir))
    max_images = cfg.detection.max_images

    # Detection model params
    model_id = cfg.detection.model_id
    conf_threshold = cfg.detection.conf_threshold
    iou_threshold = cfg.detection.iou_threshold

    out_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch 2.6+ changed weights_only default to True, which breaks ultralytics checkpoints.
    # Patch torch.load to use weights_only=False for trusted ultralytics model files.
    _original_torch_load = torch.load
    def _patched_torch_load(f, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(f, *args, **kwargs)
    torch.load = _patched_torch_load

    print(f"Inicializando modelo YOLO: {model_id}")
    model = YOLO(model_id)

    bundle_ids = load_bundle_image_ids(bundles_csv, max_images=max_images)
    print(f"Procesando {len(bundle_ids)} imagenes desde: {bundles_images_dir.resolve()}")

    for bundle_id in tqdm(bundle_ids):
        img_path = bundles_images_dir / f"{bundle_id}.jpg"
        if not img_path.exists():
            print(f"  [SKIP] No encontrada: {img_path}")
            continue

        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        if not results:
            print(f"  [NO RESULT] {bundle_id}")
            continue

        result = results[0]
        boxes = result.boxes

        render = render_result(model=model, image=str(img_path), result=result)
        render.save(out_dir / f"{bundle_id}.jpg")

    print(f"Listo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
