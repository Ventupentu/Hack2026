from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image


@dataclass
class CropItem:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    source: str


class Cropper:
    """Generate detection crops and fixed fallback multi-crops."""

    def __init__(
        self,
        min_side: int = 16,
        fallback_specs: list[tuple[float, float, float, float]] | None = None,
    ) -> None:
        self.min_side = min_side
        self.fallback_specs = fallback_specs or [
            (0.15, 0.05, 0.85, 0.95),  # center full body
            (0.2, 0.05, 0.8, 0.55),  # torso
            (0.2, 0.45, 0.8, 0.95),  # legs
            (0.0, 0.1, 0.55, 0.95),  # left side
            (0.45, 0.1, 1.0, 0.95),  # right side
        ]

    def _clip_box(self, image: Image.Image, box: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        x1 = int(max(0, min(image.width - 1, round(x1))))
        y1 = int(max(0, min(image.height - 1, round(y1))))
        x2 = int(max(0, min(image.width, round(x2))))
        y2 = int(max(0, min(image.height, round(y2))))
        if x2 - x1 < self.min_side or y2 - y1 < self.min_side:
            return None
        return (x1, y1, x2, y2)

    def from_boxes(
        self,
        image: Image.Image,
        boxes: Iterable[tuple[float, float, float, float]],
        max_crops: int,
    ) -> list[CropItem]:
        crops: list[CropItem] = []
        for box in boxes:
            clipped = self._clip_box(image, box)
            if clipped is None:
                continue
            crop = image.crop(clipped)
            crops.append(CropItem(image=crop, bbox=clipped, source="detector"))
            if len(crops) >= max_crops:
                break
        return crops

    def fallback_multi_crops(self, image: Image.Image) -> list[CropItem]:
        crops: list[CropItem] = []
        for spec in self.fallback_specs:
            x1 = int(spec[0] * image.width)
            y1 = int(spec[1] * image.height)
            x2 = int(spec[2] * image.width)
            y2 = int(spec[3] * image.height)
            clipped = self._clip_box(image, (x1, y1, x2, y2))
            if clipped is None:
                continue
            crops.append(CropItem(image=image.crop(clipped), bbox=clipped, source="fallback"))
        return crops

    def build_query_crops(
        self,
        image: Image.Image,
        detector_boxes: Iterable[tuple[float, float, float, float]],
        max_boxes: int,
        min_detector_crops_for_no_fallback: int = 2,
    ) -> list[CropItem]:
        crops = self.from_boxes(image=image, boxes=detector_boxes, max_crops=max_boxes)
        if len(crops) >= min_detector_crops_for_no_fallback:
            return crops
        fallback = self.fallback_multi_crops(image)
        return crops + fallback
