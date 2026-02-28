from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "shirt. t-shirt. polo shirt. sweater. sweatshirt. jacket. coat. trench. "
    "parka. blazer. overshirt. dress. skirt. trousers. pants. shorts. "
    "leggings. jeans. shoes. sneakers. boots. sandals. bag. handbag. "
    "backpack. belt. scarf. hat. glasses."
)


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    label: str


class GroundingDINODetector:
    """Grounding DINO wrapper for apparel box proposals."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        prompt: str = DEFAULT_PROMPT,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        max_boxes: int = 10,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_boxes = max_boxes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        LOGGER.info("Loading Grounding DINO model from %s on %s", model_id, self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _pad_and_clamp(
        boxes: torch.Tensor,
        width: int,
        height: int,
        padding_ratio: float,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes

        x1, y1, x2, y2 = boxes.unbind(dim=1)
        bw = (x2 - x1).clamp(min=1.0)
        bh = (y2 - y1).clamp(min=1.0)
        px = bw * padding_ratio
        py = bh * padding_ratio

        nx1 = (x1 - px).clamp(min=0.0, max=float(width - 1))
        ny1 = (y1 - py).clamp(min=0.0, max=float(height - 1))
        nx2 = (x2 + px).clamp(min=0.0, max=float(width - 1))
        ny2 = (y2 + py).clamp(min=0.0, max=float(height - 1))
        return torch.stack([nx1, ny1, nx2, ny2], dim=1)

    def detect(
        self,
        image: Image.Image,
        prompt: str | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        max_boxes: int | None = None,
        padding_ratio: float = 0.15,
    ) -> list[Detection]:
        """Run box detection and return padded NMS-filtered detections."""
        used_prompt = prompt or self.prompt
        used_box_th = self.box_threshold if box_threshold is None else box_threshold
        used_text_th = self.text_threshold if text_threshold is None else text_threshold
        used_max_boxes = self.max_boxes if max_boxes is None else max_boxes

        inputs = self.processor(images=image, text=used_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=used_box_th,
            text_threshold=used_text_th,
            target_sizes=target_sizes,
        )
        if not processed:
            return []

        result = processed[0]
        boxes: torch.Tensor = result.get("boxes", torch.empty((0, 4), device=self.device))
        scores: torch.Tensor = result.get("scores", torch.empty((0,), device=self.device))
        labels_raw: Iterable[str] = result.get("labels", [])

        if boxes.numel() == 0:
            return []

        keep = nms(boxes, scores, self.nms_iou_threshold)
        keep = keep[:used_max_boxes]
        keep_list = keep.tolist()

        boxes = boxes[keep]
        scores = scores[keep]
        raw_labels = [str(lbl) for lbl in labels_raw]
        labels = [
            raw_labels[idx] if idx < len(raw_labels) else "apparel"
            for idx in keep_list
        ]

        boxes = self._pad_and_clamp(boxes, image.width, image.height, padding_ratio)

        detections: list[Detection] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=float(scores[i].item()),
                    label=labels[i] if i < len(labels) else "apparel",
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:used_max_boxes]
