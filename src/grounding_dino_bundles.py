# src/grounding_dino_bundles.py
from __future__ import annotations

from pathlib import Path
from typing import List, Set

import cv2
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
import time

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

"""
Categorías encontradas: ['3/4 coat', 'accessories', 'ankle boot', 'anorak', 'athletic footwear', 'baby accesories', 'baby bermudas', 'baby body', 'baby bonnet', 'baby cardigan', 'baby dress', 'baby jacket/coat', 'baby leggings', 'baby outfit', 'baby overall', 'baby panty/underp.', 'baby polo shirt', 'baby pyjama', 'baby romper suit', 'baby shirt', 'baby skirt', 'baby socks', 'baby sweater', 'baby swimsuit', 'baby t-shirt', 'baby tracksuit', 'baby trousers', 'baby waistcoat', 'baby wind-jacket', 'bathrobe/dres.gown', 'beach sandal', 'belt', 'bermuda', 'bib overall', 'blazer', 'body lotion', 'body oil', 'bodysuit', 'books', 'boot', 'bow tie/cummerband', 'bra', 'candle', 'cardigan', 'coat', 'dress', 'eau de cologne', 'eau de parfum', 'eau de toilette', 'ensemble..set', 'eye make up', 'eyes contour', 'facial cosmetics', 'facial sunscreen', 'flat ankle boot', 'flat boot', 'flat shoes', 'glasses', 'gloves', 'hair cosmetics', 'hand bag-rucksack', 'hand cream', 'hat', 'heeled ankle boot', 'heeled boot', 'heeled shoes', 'high tops', 'home shoes', 'imit jeweller', 'knitted waistcoat', 'leggings', 'leisure and sports', 'lip balm', 'lip make up', 'lip sunscreen', 'matches', 'moccasins', 'moisturising cream', 'nail cosmetics', 'nail polish', 'newborn', 'newborn tricot', 'nightie/pyjamas', 'overall', 'overshirt', 'panty/underpant', 'parka', 'perfume', 'perfumed soap', 'polo shirt', 'powder brush-puff', 'purse wallet', 'rain boot', 'running shoes', 'sandal', 'scarf', 'shampoo', 'shawl/foulard', 'shirt', 'shoes', 'shorts', 'skirt', 'sleeveless pad. jacket', 'socks', 'sport shoes', 'sporty sandal', 'stationery', 'stockings-tights', 'suspenders', 'sweater', 'sweatshirt', 'swimsuit', 't-shirt', 'tie', 'tops and others', 'towel', 'trainers', 'trench raincoat', 'trousers', 'umbrella', 'underwear', 'uniform', 'vamp/pinky', 'waistcoat', 'wallets', 'wedge', 'wind-jacket'] ...
Número de categorías: 127
"""
MODEL_ID = "IDEA-Research/grounding-dino-base"  # or grounding-dino-tiny to go faster
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
MAX_IMAGES = 10
OUTPUT_DIR = Path("/scratch/tesla8/sgrodriguez23/dino_outputs")
MAX_TOKENS = 250  # Grounding DINO hard limit is 256 tokens

def build_classes_prompt(categories: List[str], processor, max_tokens: int = MAX_TOKENS) -> str:
    """
    Build a Grounding DINO text prompt from category list, truncating to stay
    within the model's 256-token hard limit.
    """
    included = []
    for cat in categories:
        candidate = " . ".join(included + [cat]) + " ."
        token_count = len(processor.tokenizer(candidate, add_special_tokens=True)["input_ids"])
        if token_count > max_tokens:
            break
        included.append(cat)
    prompt = " . ".join(included) + " ."
    print(f"Using {len(included)}/{len(categories)} categories in prompt ({token_count} tokens)")
    return prompt


def load_categories(product_csv: Path) -> List[str]:
    """
    Lee product_dataset.csv y toma la última columna como categorías.
    """
    df = pd.read_csv(product_csv)
    categories = (
        df.iloc[:, -1]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    return sorted(set(categories))


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
    # Rutas desde tu config Hydra
    # Asume que en cfg.files tienes:
    # - bundles_dataset
    # - product_dataset
    # - bundles_images
    bundles_csv = Path(cfg.files.bundles_dataset)
    product_csv = Path(cfg.files.product_dataset)
    bundles_images_dir = Path(cfg.files.bundles_images)

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando categorías...")
    categories = load_categories(product_csv)
    print(f"Categorías encontradas: {categories} ...")  # muestra solo las primeras 10 para no saturar
    print(f"Número de categorías: {len(categories)}")
    time.sleep(20) 
    # Prompt de texto para Grounding DINO
    # formato típico: "dress . shirt . trousers ."
    classes_prompt = " . ".join(categories) + " ."

    print("Inicializando modelo Grounding DINO...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    # Build prompt truncated to model's 256-token hard limit
    classes_prompt = build_classes_prompt(categories, processor)

    bundle_ids = load_bundle_image_ids(
        bundles_csv, max_images=cfg.get("max_images", MAX_IMAGES)  # para pruebas rápidas
    )
    print(f"Procesando {len(bundle_ids)} imágenes...")
    print(classes_prompt)
    time.sleep(10) 
    for bundle_id in tqdm(bundle_ids):
        img_path = bundles_images_dir / f"{bundle_id}.jpg"
        if not img_path.exists():
            continue

        # Predicción
        pil_image = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_image, text=classes_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        # Dibujado simple
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # results contiene: boxes (xyxy), scores, labels (texto)
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = map(int, box.tolist())
            text = f"{label} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, text, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

        cv2.imwrite(str(out_dir / f"{bundle_id}.jpg"), img)

    print(f"Listo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()