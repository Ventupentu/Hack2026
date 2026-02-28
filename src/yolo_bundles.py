# src/yolo_bundles.py
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "valentinafeve/yolos-fashionpedia"
BOX_THRESHOLD = 0.60
MAX_IMAGES = 10
OUTPUT_DIR = Path("/scratch/tesla8/sgrodriguez23/yolo_outputs")

# YOLOS-Fashionpedia labels that map to our product categories.
# Parts-of-garment labels (sleeve, collar, neckline, pocket, etc.) are excluded.
KEEP_LABELS: set[str] = {
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit",
    "cape",
    "glasses",
    "hat",
    "headband, head covering, hair accessory",
    "tie",
    "glove",
    "watch",
    "belt",
    "leg warmer",
    "tights, stockings",
    "sock",
    "shoe",
    "bag, wallet",
    "scarf",
    "umbrella",
}


def load_bundle_image_ids(bundles_csv: Path, max_images: int | None = None) -> List[str]:
    """
    Lee bundles_dataset.csv y devuelve bundle_asset_id para localizar imágenes locales.
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

    print("Inicializando modelo YOLOS-Fashionpedia...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # Mapeo id -> label del propio modelo
    id2label = model.config.id2label
    print(f"Categorías del modelo: {list(id2label.values())}")

    bundle_ids = load_bundle_image_ids(
        bundles_csv, max_images=cfg.get("max_images", MAX_IMAGES)
    )
    print(f"Procesando {len(bundle_ids)} imágenes desde: {bundles_images_dir.resolve()}")

    for bundle_id in tqdm(bundle_ids):
        img_path = bundles_images_dir / f"{bundle_id}.jpg"
        if not img_path.exists():
            print(f"  [SKIP] No encontrada: {img_path}")
            continue

        pil_image = Image.open(img_path).convert("RGB")

        # Preprocesado y forward
        inputs = feature_extractor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-proceso: convierte logits/boxes a coordenadas absolutas
        target_sizes = torch.tensor([pil_image.size[::-1]], device=device)  # (H, W)
        results = feature_extractor.post_process_object_detection(
            outputs,
            threshold=BOX_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        # Dibujado
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for box, score, label_id in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = id2label[label_id.item()]
            if label not in KEEP_LABELS:
                continue
            text = f"{label} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, text, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

        cv2.imwrite(str(out_dir / f"{bundle_id}.jpg"), img)

    print(f"Listo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()