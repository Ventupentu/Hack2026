#!/usr/bin/env python3
"""Generate offline augmentations for bundle/product retrieval datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError
try:
    from torchvision.transforms import functional as TF
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    TF = None


LOGGER = logging.getLogger("offline_augment")


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    image_path: Path


@dataclass(frozen=True)
class AugmentTask:
    split_name: str  # "bundle" or "product"
    asset_id: str
    original_path: Path
    out_dir: Path
    num_augs: int


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline augmentation for fashion retrieval.")
    parser.add_argument("--bundles_manifest", type=Path, required=True)
    parser.add_argument("--products_manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--products_images_dir",
        type=Path,
        default=None,
        help=(
            "Optional local products image dir used when products manifest has URLs or "
            "no image_path. Expected filenames: <product_asset_id>.jpg/.jpeg/.png/.webp"
        ),
    )

    parser.add_argument("--bundles_num_augs", type=int, default=4)
    parser.add_argument("--products_num_augs", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jpeg_min_quality", type=int, default=60)
    parser.add_argument("--jpeg_max_quality", type=int, default=95)
    parser.add_argument("--disable_product_flip", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(*parts: str | int) -> int:
    text = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def read_manifest_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Invalid JSON at %s:%d (%s)", path, line_no, exc)
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    raise ValueError(f"Unsupported manifest extension for {path}. Use .jsonl or .csv")


def first_non_empty(row: Dict[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        text = str(value).strip() if value is not None else ""
        if text and text.lower() not in {"nan", "none"}:
            return text
    return ""


def load_assets(
    manifest_path: Path,
    id_keys: Sequence[str],
    image_keys: Sequence[str],
    fallback_image_dir: Optional[Path] = None,
) -> List[AssetRecord]:
    rows = read_manifest_rows(manifest_path)
    assets: List[AssetRecord] = []
    seen_ids: set[str] = set()
    skipped = 0

    for row in rows:
        asset_id = first_non_empty(row, id_keys)
        image_path = first_non_empty(row, image_keys)
        if image_path.lower().startswith(("http://", "https://")):
            image_path = ""

        if not image_path and asset_id and fallback_image_dir is not None:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = fallback_image_dir / f"{asset_id}{ext}"
                if candidate.exists():
                    image_path = str(candidate)
                    break

        if not asset_id or not image_path:
            skipped += 1
            continue
        if asset_id in seen_ids:
            continue
        seen_ids.add(asset_id)
        assets.append(AssetRecord(asset_id=asset_id, image_path=Path(image_path).expanduser().resolve()))

    if skipped:
        LOGGER.warning("%s: skipped %d rows with missing id/path.", manifest_path, skipped)

    assets.sort(key=lambda item: item.asset_id)
    return assets


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def open_image_rgb(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        LOGGER.warning("Cannot read image %s (%s)", path, exc)
        return None


def sample_resized_crop_params(
    width: int,
    height: int,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    rng: random.Random,
) -> Tuple[int, int, int, int]:
    area = width * height
    log_ratio_min = math.log(ratio[0])
    log_ratio_max = math.log(ratio[1])

    for _ in range(10):
        target_area = rng.uniform(scale[0], scale[1]) * area
        aspect = math.exp(rng.uniform(log_ratio_min, log_ratio_max))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            y = rng.randint(0, height - crop_h)
            x = rng.randint(0, width - crop_w)
            return y, x, crop_h, crop_w

    in_ratio = width / float(height)
    if in_ratio < ratio[0]:
        crop_w = width
        crop_h = int(round(crop_w / ratio[0]))
    elif in_ratio > ratio[1]:
        crop_h = height
        crop_w = int(round(crop_h * ratio[1]))
    else:
        crop_w = width
        crop_h = height

    y = max((height - crop_h) // 2, 0)
    x = max((width - crop_w) // 2, 0)
    crop_h = min(crop_h, height)
    crop_w = min(crop_w, width)
    return y, x, crop_h, crop_w


def apply_color_jitter(
    image: Image.Image,
    rng: random.Random,
    brightness: Tuple[float, float],
    contrast: Tuple[float, float],
    saturation: Tuple[float, float],
    hue: Tuple[float, float],
) -> Tuple[Image.Image, str]:
    values = {
        "b": rng.uniform(brightness[0], brightness[1]),
        "c": rng.uniform(contrast[0], contrast[1]),
        "s": rng.uniform(saturation[0], saturation[1]),
        "h": rng.uniform(hue[0], hue[1]),
    }

    operations = list(values.keys())
    rng.shuffle(operations)

    for op in operations:
        if op == "b":
            image = TF.adjust_brightness(image, values["b"])
        elif op == "c":
            image = TF.adjust_contrast(image, values["c"])
        elif op == "s":
            image = TF.adjust_saturation(image, values["s"])
        else:
            image = TF.adjust_hue(image, values["h"])

    recipe = (
        f"jitter(b={values['b']:.3f},c={values['c']:.3f},"
        f"s={values['s']:.3f},h={values['h']:.3f})"
    )
    return image, recipe


def apply_cutout(
    image: Image.Image,
    rng: random.Random,
    probability: float = 0.25,
    min_area: float = 0.02,
    max_area: float = 0.12,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> Tuple[Image.Image, Optional[str]]:
    if rng.random() >= probability:
        return image, None

    width, height = image.size
    area = width * height
    for _ in range(10):
        target = rng.uniform(min_area, max_area) * area
        aspect = rng.uniform(min_ratio, max_ratio)
        erase_w = int(round(math.sqrt(target * aspect)))
        erase_h = int(round(math.sqrt(target / aspect)))
        if 0 < erase_w < width and 0 < erase_h < height:
            x = rng.randint(0, width - erase_w)
            y = rng.randint(0, height - erase_h)
            fill = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            drawer = ImageDraw.Draw(image)
            drawer.rectangle((x, y, x + erase_w, y + erase_h), fill=fill)
            return image, f"cutout(x={x},y={y},w={erase_w},h={erase_h})"

    return image, None


def augment_bundle(
    base_image: Image.Image,
    img_size: int,
    rng: random.Random,
    jpeg_min_quality: int,
    jpeg_max_quality: int,
) -> Tuple[Image.Image, str, int]:
    recipe_parts: List[str] = []

    y, x, crop_h, crop_w = sample_resized_crop_params(
        width=base_image.size[0],
        height=base_image.size[1],
        scale=(0.5, 1.0),
        ratio=(0.75, 1.33),
        rng=rng,
    )
    image = TF.resized_crop(base_image, y, x, crop_h, crop_w, size=[img_size, img_size], antialias=True)
    recipe_parts.append(f"rrc(y={y},x={x},h={crop_h},w={crop_w})")

    if rng.random() < 0.5:
        image = TF.hflip(image)
        recipe_parts.append("hflip")

    image, jitter_recipe = apply_color_jitter(
        image,
        rng=rng,
        brightness=(0.75, 1.25),
        contrast=(0.75, 1.25),
        saturation=(0.75, 1.25),
        hue=(-0.04, 0.04),
    )
    recipe_parts.append(jitter_recipe)

    if rng.random() < 0.20:
        radius = rng.uniform(0.1, 1.3)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        recipe_parts.append(f"gblur(r={radius:.2f})")

    image, cutout_recipe = apply_cutout(image, rng=rng, probability=0.25)
    if cutout_recipe:
        recipe_parts.append(cutout_recipe)

    jpeg_quality = rng.randint(jpeg_min_quality, jpeg_max_quality)
    recipe_parts.append(f"jpeg_q={jpeg_quality}")
    return image, ";".join(recipe_parts), jpeg_quality


def augment_product(
    base_image: Image.Image,
    img_size: int,
    rng: random.Random,
    jpeg_min_quality: int,
    jpeg_max_quality: int,
    allow_flip: bool,
) -> Tuple[Image.Image, str, int]:
    recipe_parts: List[str] = []

    y, x, crop_h, crop_w = sample_resized_crop_params(
        width=base_image.size[0],
        height=base_image.size[1],
        scale=(0.85, 1.0),
        ratio=(0.9, 1.1),
        rng=rng,
    )
    image = TF.resized_crop(base_image, y, x, crop_h, crop_w, size=[img_size, img_size], antialias=True)
    recipe_parts.append(f"rrc(y={y},x={x},h={crop_h},w={crop_w})")

    if allow_flip and rng.random() < 0.10:
        image = TF.hflip(image)
        recipe_parts.append("hflip")

    image, jitter_recipe = apply_color_jitter(
        image,
        rng=rng,
        brightness=(0.90, 1.10),
        contrast=(0.90, 1.10),
        saturation=(0.90, 1.10),
        hue=(-0.015, 0.015),
    )
    recipe_parts.append(jitter_recipe)

    if rng.random() < 0.08:
        radius = rng.uniform(0.05, 0.60)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        recipe_parts.append(f"gblur(r={radius:.2f})")

    jpeg_quality = rng.randint(jpeg_min_quality, jpeg_max_quality)
    recipe_parts.append(f"jpeg_q={jpeg_quality}")
    return image, ";".join(recipe_parts), jpeg_quality


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_task(
    task: AugmentTask,
    seed: int,
    img_size: int,
    jpeg_min_quality: int,
    jpeg_max_quality: int,
    disable_product_flip: bool,
    overwrite: bool,
) -> List[Dict[str, object]]:
    ensure_dir(task.out_dir)
    output_paths = [task.out_dir / f"{task.asset_id}__aug{i}.jpg" for i in range(task.num_augs)]

    if not overwrite and output_paths and all(path.exists() for path in output_paths):
        return [
            {
                "asset_id": task.asset_id,
                "original_path": str(task.original_path),
                "aug_path": str(path),
                "aug_index": idx,
                "aug_recipe": "existing_file",
            }
            for idx, path in enumerate(output_paths)
        ]

    base_image = open_image_rgb(task.original_path)
    if base_image is None:
        return []

    rows: List[Dict[str, object]] = []
    for aug_idx, out_path in enumerate(output_paths):
        if out_path.exists() and not overwrite:
            rows.append(
                {
                    "asset_id": task.asset_id,
                    "original_path": str(task.original_path),
                    "aug_path": str(out_path),
                    "aug_index": aug_idx,
                    "aug_recipe": "existing_file",
                }
            )
            continue

        local_rng = random.Random(stable_seed(seed, task.split_name, task.asset_id, aug_idx))

        try:
            if task.split_name == "bundle":
                aug_image, aug_recipe, quality = augment_bundle(
                    base_image.copy(),
                    img_size=img_size,
                    rng=local_rng,
                    jpeg_min_quality=jpeg_min_quality,
                    jpeg_max_quality=jpeg_max_quality,
                )
            else:
                aug_image, aug_recipe, quality = augment_product(
                    base_image.copy(),
                    img_size=img_size,
                    rng=local_rng,
                    jpeg_min_quality=jpeg_min_quality,
                    jpeg_max_quality=jpeg_max_quality,
                    allow_flip=not disable_product_flip,
                )

            aug_image.save(out_path, format="JPEG", quality=quality, optimize=True)
            rows.append(
                {
                    "asset_id": task.asset_id,
                    "original_path": str(task.original_path),
                    "aug_path": str(out_path),
                    "aug_index": aug_idx,
                    "aug_recipe": aug_recipe,
                }
            )
        except Exception as exc:
            LOGGER.warning(
                "Failed augmentation for %s=%s aug=%d (%s)",
                task.split_name,
                task.asset_id,
                aug_idx,
                exc,
            )

    return rows


def run_split(
    split_name: str,
    assets: Sequence[AssetRecord],
    out_dir: Path,
    num_augs: int,
    seed: int,
    img_size: int,
    jpeg_min_quality: int,
    jpeg_max_quality: int,
    workers: int,
    disable_product_flip: bool,
    overwrite: bool,
) -> List[Dict[str, object]]:
    tasks = [
        AugmentTask(
            split_name=split_name,
            asset_id=item.asset_id,
            original_path=item.image_path,
            out_dir=out_dir,
            num_augs=num_augs,
        )
        for item in assets
    ]

    all_rows: List[Dict[str, object]] = []
    total = len(tasks)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_task,
                task=task,
                seed=seed,
                img_size=img_size,
                jpeg_min_quality=jpeg_min_quality,
                jpeg_max_quality=jpeg_max_quality,
                disable_product_flip=disable_product_flip,
                overwrite=overwrite,
            )
            for task in tasks
        ]

        for future in as_completed(futures):
            done += 1
            try:
                all_rows.extend(future.result())
            except Exception as exc:
                LOGGER.warning("Unexpected worker failure in %s split (%s)", split_name, exc)

            if done % 500 == 0 or done == total:
                LOGGER.info("%s progress: %d/%d", split_name, done, total)

    all_rows.sort(key=lambda row: (str(row["asset_id"]), int(row["aug_index"])))
    return all_rows


def main() -> None:
    setup_logging()
    args = parse_args()

    if TF is None:
        raise ModuleNotFoundError(
            "Missing dependency 'torchvision'. Install with:\n"
            "  pip install torchvision"
        )

    if args.bundles_num_augs < 0 or args.products_num_augs < 0:
        raise ValueError("--bundles_num_augs and --products_num_augs must be >= 0")
    if args.img_size <= 0:
        raise ValueError("--img_size must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.jpeg_min_quality < 1 or args.jpeg_max_quality > 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    if args.jpeg_min_quality > args.jpeg_max_quality:
        raise ValueError("--jpeg_min_quality must be <= --jpeg_max_quality")

    bundles = load_assets(
        manifest_path=args.bundles_manifest,
        id_keys=("bundle_asset_id", "bundle_id", "asset_id", "id"),
        image_keys=("image_path", "bundle_image_path", "bundle_path", "path"),
    )
    products_images_dir: Optional[Path]
    if args.products_images_dir is None:
        products_images_dir = None
        for candidate in (
            args.products_manifest.parent / "product_images",
            Path("data/product_images"),
        ):
            if candidate.exists():
                products_images_dir = candidate.resolve()
                break
    else:
        products_images_dir = args.products_images_dir.expanduser().resolve()

    products = load_assets(
        manifest_path=args.products_manifest,
        id_keys=("product_asset_id", "product_id", "asset_id", "id"),
        image_keys=("image_path", "product_image_path", "product_path", "path", "product_image_url"),
        fallback_image_dir=products_images_dir,
    )

    LOGGER.info("Loaded bundle assets: %d", len(bundles))
    LOGGER.info("Loaded product assets: %d", len(products))

    out_dir = args.out_dir.expanduser().resolve()
    bundles_aug_dir = out_dir / "bundles_aug"
    products_aug_dir = out_dir / "products_aug"
    ensure_dir(bundles_aug_dir)
    ensure_dir(products_aug_dir)

    bundle_rows = run_split(
        split_name="bundle",
        assets=bundles,
        out_dir=bundles_aug_dir,
        num_augs=args.bundles_num_augs,
        seed=args.seed,
        img_size=args.img_size,
        jpeg_min_quality=args.jpeg_min_quality,
        jpeg_max_quality=args.jpeg_max_quality,
        workers=args.workers,
        disable_product_flip=args.disable_product_flip,
        overwrite=args.overwrite,
    )

    product_rows = run_split(
        split_name="product",
        assets=products,
        out_dir=products_aug_dir,
        num_augs=args.products_num_augs,
        seed=args.seed,
        img_size=args.img_size,
        jpeg_min_quality=args.jpeg_min_quality,
        jpeg_max_quality=args.jpeg_max_quality,
        workers=args.workers,
        disable_product_flip=args.disable_product_flip,
        overwrite=args.overwrite,
    )

    bundles_manifest_out = out_dir / "bundles_aug_manifest.jsonl"
    products_manifest_out = out_dir / "products_aug_manifest.jsonl"
    write_jsonl(bundles_manifest_out, bundle_rows)
    write_jsonl(products_manifest_out, product_rows)

    LOGGER.info("Saved %d bundle aug rows -> %s", len(bundle_rows), bundles_manifest_out)
    LOGGER.info("Saved %d product aug rows -> %s", len(product_rows), products_manifest_out)


if __name__ == "__main__":
    main()
