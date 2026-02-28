from __future__ import annotations

from pathlib import Path
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from ultralyticsplus import YOLO, render_result

MODEL_ID = "kesimeg/yolov8n-clothing-detection"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_IMAGES = 10
OUTPUT_DIR = Path("/scratch/tesla8/sgrodriguez23/yolov8_clothing_outputs")


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
def main(cfg: DictConfig) -> None:
    bundles_csv = Path(cfg.files.bundles_dataset)
    bundles_images_dir = Path(cfg.files.bundles_images)

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_threshold = float(cfg.get("conf_threshold", CONF_THRESHOLD))
    iou_threshold = float(cfg.get("iou_threshold", IOU_THRESHOLD))

    print(f"Inicializando modelo YOLO: {MODEL_ID}")
    model = YOLO(MODEL_ID)

    bundle_ids = load_bundle_image_ids(
        bundles_csv, max_images=cfg.get("max_images", MAX_IMAGES)
    )
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
        print(f"[{bundle_id}] boxes: {boxes}")

        render = render_result(model=model, image=str(img_path), result=result)
        render.save(out_dir / f"{bundle_id}.jpg")

    print(f"Listo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
