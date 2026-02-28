from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

try:
    from ultralyticsplus import YOLO, render_result
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime
    YOLO = None  # type: ignore[assignment]
    render_result = None  # type: ignore[assignment]

MODEL_ID = "kesimeg/yolov8n-clothing-detection"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_BOXES_PER_IMAGE = 15
MIN_AREA_RATIO = 0.001
MAX_IMAGES = 10
OUTPUT_DIR = Path("outputs/yolov8_clothing")

BoxXYXY = Tuple[int, int, int, int]
ScoredBox = Tuple[int, int, int, int, float]


def load_bundle_image_ids(bundles_csv: Path, max_images: int | None = None) -> List[str]:
    """Lee bundles_dataset.csv y devuelve bundle_asset_id para localizar imagenes locales."""
    df = pd.read_csv(bundles_csv)
    ids = df["bundle_asset_id"].astype(str).tolist()
    if max_images is not None:
        ids = ids[:max_images]
    return ids


def _sanitize_box(
    xyxy: List[float],
    image_w: int,
    image_h: int,
    min_area_ratio: float,
) -> Optional[BoxXYXY]:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(1, min(x2, image_w))
    y2 = max(1, min(y2, image_h))
    if x2 <= x1 or y2 <= y1:
        return None

    area = (x2 - x1) * (y2 - y1)
    if area < int(image_w * image_h * min_area_ratio):
        return None
    return (x1, y1, x2, y2)


def extract_boxes_from_result(
    result: Any,
    max_boxes_per_image: int = MAX_BOXES_PER_IMAGE,
    min_area_ratio: float = MIN_AREA_RATIO,
) -> List[ScoredBox]:
    """Convierte output de YOLO a boxes XYXY enteras y ordenadas por confianza."""
    raw_boxes = getattr(result, "boxes", None)
    if raw_boxes is None or len(raw_boxes) == 0:
        return []

    orig_h, orig_w = result.orig_shape[:2]
    xyxy = raw_boxes.xyxy.detach().cpu().tolist()
    conf = raw_boxes.conf.detach().cpu().tolist() if hasattr(raw_boxes, "conf") else [1.0] * len(xyxy)

    scored_boxes: List[ScoredBox] = []
    for coords, score in zip(xyxy, conf):
        clean_box = _sanitize_box(coords, image_w=orig_w, image_h=orig_h, min_area_ratio=min_area_ratio)
        if clean_box is None:
            continue
        scored_boxes.append((*clean_box, float(score)))

    scored_boxes.sort(key=lambda box: box[4], reverse=True)
    return scored_boxes[:max_boxes_per_image]


@dataclass
class ClothingYOLODetector:
    """Detector reutilizable para extraer boxes de prendas por imagen."""

    model_id: str = MODEL_ID
    conf_threshold: float = CONF_THRESHOLD
    iou_threshold: float = IOU_THRESHOLD
    max_boxes_per_image: int = MAX_BOXES_PER_IMAGE
    min_area_ratio: float = MIN_AREA_RATIO

    def __post_init__(self) -> None:
        if YOLO is None:
            raise ModuleNotFoundError(
                "ultralyticsplus no esta instalado. Instala con: pip install ultralyticsplus"
            )
        self.model = YOLO(self.model_id)

    def detect_boxes(self, image_path: Path) -> List[ScoredBox]:
        """Devuelve boxes [x1, y1, x2, y2, score] para una imagen."""
        results = self.model.predict(
            str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not results:
            return []
        return extract_boxes_from_result(
            results[0],
            max_boxes_per_image=self.max_boxes_per_image,
            min_area_ratio=self.min_area_ratio,
        )

    def detect_boxes_without_scores(self, image_path: Path) -> List[BoxXYXY]:
        """Devuelve boxes [x1, y1, x2, y2] para uso directo en recorte."""
        return [box[:4] for box in self.detect_boxes(image_path)]


def detect_boxes_for_assets(
    detector: ClothingYOLODetector,
    asset_to_image: Dict[str, Path],
    show_progress: bool = True,
) -> Dict[str, List[BoxXYXY]]:
    """Ejecuta deteccion para todos los assets y devuelve map asset_id -> boxes."""
    iterator: Iterable[Tuple[str, Path]] = asset_to_image.items()
    if show_progress:
        iterator = tqdm(asset_to_image.items(), total=len(asset_to_image), desc="Detecting boxes")

    out: Dict[str, List[BoxXYXY]] = {}
    for asset_id, image_path in iterator:
        if not image_path.exists():
            out[asset_id] = []
            continue
        out[asset_id] = detector.detect_boxes_without_scores(image_path)
    return out


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    files_cfg = cfg.get("files", {})
    bundles_csv = Path(files_cfg.get("bundles_dataset", "data/bundles_dataset.csv"))
    bundles_images_dir = Path(files_cfg.get("bundles_images", "data/bundle_images"))

    out_dir = Path(cfg.get("detection_output_dir", str(OUTPUT_DIR)))
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_threshold = float(cfg.get("conf_threshold", CONF_THRESHOLD))
    iou_threshold = float(cfg.get("iou_threshold", IOU_THRESHOLD))
    max_boxes_per_image = int(cfg.get("max_boxes_per_image", MAX_BOXES_PER_IMAGE))
    min_area_ratio = float(cfg.get("min_area_ratio", MIN_AREA_RATIO))

    print(f"Inicializando modelo YOLO: {MODEL_ID}")
    detector = ClothingYOLODetector(
        model_id=MODEL_ID,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_boxes_per_image=max_boxes_per_image,
        min_area_ratio=min_area_ratio,
    )

    bundle_ids = load_bundle_image_ids(
        bundles_csv, max_images=cfg.get("max_images", MAX_IMAGES)
    )
    print(f"Procesando {len(bundle_ids)} imagenes desde: {bundles_images_dir.resolve()}")

    boxes_json: Dict[str, List[List[float]]] = {}
    for bundle_id in tqdm(bundle_ids):
        img_path = bundles_images_dir / f"{bundle_id}.jpg"
        if not img_path.exists():
            print(f"  [SKIP] No encontrada: {img_path}")
            continue

        scored_boxes = detector.detect_boxes(img_path)
        boxes_json[bundle_id] = [list(box) for box in scored_boxes]
        print(f"[{bundle_id}] num_boxes={len(scored_boxes)}")

        if render_result is not None:
            results = detector.model.predict(
                str(img_path),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            if results:
                render = render_result(model=detector.model, image=str(img_path), result=results[0])
                render.save(out_dir / f"{bundle_id}.jpg")

    (out_dir / "bundle_boxes.json").write_text(
        json.dumps(boxes_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Listo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
