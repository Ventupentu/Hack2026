from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

import torch
import functools

# PyTorch 2.6 changed torch.load default to weights_only=True,
# which breaks ultralytics YOLO model loading. Patch it back.
_orig_torch_load = torch.load
@functools.wraps(_orig_torch_load)
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load  # type: ignore[assignment]

try:
    from ultralyticsplus import YOLO, render_result
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore[assignment]
    render_result = None  # type: ignore[assignment]


MODEL_ID = "kesimeg/yolov8n-clothing-detection"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_BOXES_PER_IMAGE = 15
MIN_AREA_RATIO = 0.001

BoxXYXY = Tuple[int, int, int, int]
ScoredBox = Tuple[int, int, int, int, float]


def _patch_torch_load_weights_only_false() -> None:
    """PyTorch 2.6 changed torch.load default weights_only=True; YOLO checkpoints need False."""
    import torch

    if getattr(torch.load, "__name__", "") == "_patched_torch_load":
        return

    original_torch_load = torch.load

    def _patched_torch_load(f, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(f, *args, **kwargs)

    torch.load = _patched_torch_load


def load_bundle_image_ids(bundles_csv: Path, max_images: int | None = None) -> List[str]:
    """Read bundles CSV and return bundle asset ids."""
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
    """Convert YOLO output to sorted XYXY boxes with confidence."""
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
    """Reusable detector for bundle clothing regions."""

    model_id: str = MODEL_ID
    conf_threshold: float = CONF_THRESHOLD
    iou_threshold: float = IOU_THRESHOLD
    max_boxes_per_image: int = MAX_BOXES_PER_IMAGE
    min_area_ratio: float = MIN_AREA_RATIO

    def __post_init__(self) -> None:
        if YOLO is None:
            raise ModuleNotFoundError(
                "ultralyticsplus is not installed. Install with: pip install ultralyticsplus"
            )
        _patch_torch_load_weights_only_false()
        self.model = YOLO(self.model_id)

    def detect_boxes(self, image_path: Path) -> List[ScoredBox]:
        """Return [x1, y1, x2, y2, score] per detected box."""
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
        """Return [x1, y1, x2, y2] for direct cropping."""
        return [box[:4] for box in self.detect_boxes(image_path)]


def detect_boxes_for_assets(
    detector: ClothingYOLODetector,
    asset_to_image: Dict[str, Path],
    show_progress: bool = True,
) -> Dict[str, List[BoxXYXY]]:
    """Run detection for all assets and return asset_id -> boxes."""
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