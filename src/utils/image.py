from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger(__name__)


def load_image_rgb(path: str | Path) -> Image.Image | None:
    """Load RGB image, handling corrupted files gracefully."""
    p = Path(path)
    if not p.exists():
        LOGGER.warning("Image path does not exist: %s", p)
        return None
    try:
        with Image.open(p) as img:
            return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to open image %s: %s", p, exc)
        return None
