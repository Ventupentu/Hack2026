"""Top-1-per-crop inference script.

Strategy: each YOLO-detected crop maps to exactly ONE product (its best match).
This naturally yields ~4-5 products per bundle (matching actual garment count),
avoiding the precision loss of always returning 15 candidates.

Usage:
    python -m src.infer_top1 \
        infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt \
        infer.val_ratio=0.1
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from src.config import InditexConfig
from src.new_infer import (
    ProductIndex,
    build_crop_items,
    build_image_map,
    collate_skip_none,
    detect_bundles,
    encode_bundle_crops,
    encode_products,
    filter_cross_gender,
    load_bundle_genders,
    load_openclip_model,
    load_product_categories,
    load_product_genders,
    parse_ks,
    resolve_device,
    set_seed,
)
from src.utils.metrics import evaluate_bundle_retrieval

_GENDER_UNKNOWN = 0


# ---------------------------------------------------------------------------
# Top-1-per-crop retrieval
# ---------------------------------------------------------------------------

def retrieve_top1_per_crop(
    bundle_crops: Dict[str, List[Tuple[np.ndarray, float]]],
    product_index: ProductIndex,
    bundle_to_gender: Optional[Dict[str, int]] = None,
    product_to_gender: Optional[Dict[str, int]] = None,
    gender_filter: bool = True,
    score_threshold: float = 0.0,
    skip_full_image: bool = True,
) -> Dict[str, List[str]]:
    """Top-1-per-crop retrieval: each crop contributes exactly one product.

    For each crop embedding in a bundle:
      1. Search top-1 from the product index
      2. Apply score threshold
      3. Apply gender filter
      4. Deduplicate: if two crops match the same product, keep it once

    When `skip_full_image` is True (default), crops with confidence == 0.6
    (the full-image fallback added alongside YOLO detections) are skipped.
    Full-image crops with confidence == 1.0 (no YOLO detections) are kept.
    """
    results: Dict[str, List[str]] = {}

    for bundle_id, crops in tqdm(bundle_crops.items(), desc="Top-1 retrieval", leave=False):
        # Determine if this bundle has real YOLO detections
        has_detections = any(conf != 0.6 and conf != 1.0 for _, conf in crops)
        # Also check: if there's a conf=0.6 AND other crops, it means detections exist
        if not has_detections:
            has_detections = len(crops) > 1  # hflip TTA doubles items

        seen_products: set = set()
        matched: List[Tuple[str, float]] = []

        for emb, conf in crops:
            # Skip full-image fallback (conf=0.6) when real detections exist
            if skip_full_image and conf == 0.6:
                continue

            indices, scores = product_index.search(emb, 1)
            idx = int(indices[0, 0])
            sim = float(scores[0, 0])

            if idx < 0:
                continue

            # Score threshold
            combined = sim * max(conf, 0.1)
            if score_threshold > 0 and combined < score_threshold:
                continue

            pid = product_index.product_ids[idx]
            if pid not in seen_products:
                seen_products.add(pid)
                matched.append((pid, combined))

        # Sort by score descending
        matched.sort(key=lambda x: x[1], reverse=True)

        # Gender filter
        if gender_filter and bundle_to_gender and product_to_gender:
            bundle_gender = bundle_to_gender.get(bundle_id, _GENDER_UNKNOWN)
            matched = filter_cross_gender(matched, bundle_gender, product_to_gender)

        # Cap at 6 to match typical bundle size (4-5 garments)
        results[bundle_id] = [pid for pid, _ in matched[:6]]

    return results


# ---------------------------------------------------------------------------
# Build crop items WITHOUT full-image fallback when detections exist
# ---------------------------------------------------------------------------

def build_crop_items_no_fallback(
    bundle_ids: List[str],
    bundle_image_map: Dict[str, Path],
    bundle_to_boxes: Dict[str, list],
) -> List[Tuple[str, Path, Optional[Any], float]]:
    """Like build_crop_items but does NOT add a full-image crop when
    YOLO detections exist. Only uses full image when there are zero detections.
    """
    from src.new_infer import BoxXYXY

    items = []
    box_pad = 0.15

    for bid in bundle_ids:
        path = bundle_image_map.get(bid)
        if path is None or not path.exists():
            continue
        boxes = bundle_to_boxes.get(bid, [])
        if boxes:
            for x1, y1, x2, y2, conf in boxes:
                w, h = x2 - x1, y2 - y1
                px, py = int(w * box_pad), int(h * box_pad)
                padded_box = (x1 - px, y1 - py, x2 + px, y2 + py)
                items.append((bid, path, padded_box, conf))
            # NO full-image fallback here — we trust the detections
        else:
            # No detections → full image as sole crop
            items.append((bid, path, None, 1.0))

    return items


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    bundle_ids: List[str],
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    ks: List[int],
) -> Dict[str, float]:
    pred_ids = []
    pred_lists = []
    for bid in bundle_ids:
        if bid in predictions and bid in ground_truth:
            pred_ids.append(bid)
            pred_lists.append(predictions[bid])
    return evaluate_bundle_retrieval(pred_ids, pred_lists, ground_truth, ks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: InditexConfig) -> None:
    # ---- Paths ----
    data_dir = Path(to_absolute_path(cfg.files.data_dir))
    bundles_csv = data_dir / "bundles_dataset.csv"
    products_csv = data_dir / "product_dataset.csv"
    train_csv = data_dir / "bundles_product_match_train.csv"
    test_csv = data_dir / "bundles_product_match_test.csv"
    bundle_images_dir = Path(to_absolute_path(cfg.files.bundles_images))
    product_images_dir = Path(to_absolute_path(cfg.files.products_images))
    output_dir = Path(to_absolute_path("outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Params ----
    seed = cfg.params.seed
    device = resolve_device(cfg.params.device)
    batch_size = cfg.params.batch_size
    num_workers = cfg.params.num_workers
    amp = cfg.params.amp
    val_ratio = cfg.infer.val_ratio
    eval_ks = parse_ks(cfg.infer.eval_ks)
    checkpoint_path = cfg.infer.checkpoint_path
    n_tta = cfg.infer.tta_num_augs

    # Post-processing params
    gender_filter = bool(getattr(cfg.infer, "gender_filter", True))
    score_threshold = float(getattr(cfg.infer, "score_threshold", 0.0))

    set_seed(seed)

    # ---- Load model ----
    print("=" * 60)
    print("INFER TOP-1 — One product per detected crop")
    print("=" * 60)
    model, preprocess, tokenizer = load_openclip_model(checkpoint_path, device)
    print(f"Device: {device} | AMP: {amp} | TTA: {n_tta}")

    # ---- Load data ----
    bundles_df = pd.read_csv(bundles_csv)
    products_df = pd.read_csv(products_csv)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    bundle_image_map = build_image_map(bundle_images_dir)
    product_image_map = build_image_map(product_images_dir)

    all_product_ids = products_df["product_asset_id"].astype(str).tolist()
    product_ids = [pid for pid in all_product_ids if pid in product_image_map]
    product_to_text: Dict[str, str] = dict(zip(
        products_df["product_asset_id"].astype(str),
        products_df["product_description"].fillna("").astype(str),
    ))
    print(f"Products: {len(product_ids)} with images / {len(all_product_ids)} total")

    # ---- Load gender maps ----
    bundle_to_gender = load_bundle_genders(bundles_df)
    products_gender_csv = data_dir / "product_dataset_with_gender.csv"
    product_to_gender = load_product_genders(products_gender_csv)

    if gender_filter and bundle_to_gender and product_to_gender:
        print(f"Gender filter: {len(bundle_to_gender)} bundles, {len(product_to_gender)} products")
    else:
        print("Gender filter disabled.")
    print(f"Score threshold: {score_threshold}")

    # ---- Encode products (with TTA) ----
    print(f"\n[1/4] Encoding products (TTA={n_tta})...")
    encoded_pids, product_embeddings = encode_products(
        product_ids, product_image_map, product_to_text,
        model, preprocess, tokenizer, device,
        batch_size, num_workers, amp, n_tta,
    )
    print(f"  Product embeddings: {product_embeddings.shape}")

    # ---- Build search index ----
    print("\n[2/4] Building search index...")
    search_index = ProductIndex(encoded_pids, product_embeddings)

    # ---- Validation ----
    if val_ratio > 0:
        print(f"\n[3/4] Validation (val_ratio={val_ratio})...")
        train_bundle_ids = train_df["bundle_asset_id"].astype(str).tolist()
        unique_train_bundles = list(dict.fromkeys(train_bundle_ids))
        random.Random(seed).shuffle(unique_train_bundles)
        val_size = max(1, int(round(len(unique_train_bundles) * val_ratio)))
        val_bundle_ids = sorted(unique_train_bundles[:val_size])
        print(f"  Val bundles: {len(val_bundle_ids)}")

        # Ground truth
        gt_map: Dict[str, Set[str]] = defaultdict(set)
        for row in train_df.itertuples(index=False):
            gt_map[str(row.bundle_asset_id)].add(str(row.product_asset_id))

        # Detect + encode + retrieve for val bundles
        val_boxes = detect_bundles(val_bundle_ids, bundle_image_map, cfg.params)
        val_crop_items = build_crop_items_no_fallback(
            val_bundle_ids, bundle_image_map, val_boxes,
        )
        val_crop_embs = encode_bundle_crops(
            val_crop_items, model, preprocess, device, batch_size, num_workers, amp,
        )
        val_predictions = retrieve_top1_per_crop(
            val_crop_embs, search_index,
            bundle_to_gender=bundle_to_gender,
            product_to_gender=product_to_gender,
            gender_filter=gender_filter,
            score_threshold=score_threshold,
        )

        # Print stats about predictions
        num_preds = [len(v) for v in val_predictions.values()]
        print(f"  Avg products/bundle: {np.mean(num_preds):.1f} "
              f"(min={min(num_preds)}, max={max(num_preds)}, "
              f"median={np.median(num_preds):.0f})")

        val_metrics = validate(val_bundle_ids, val_predictions, gt_map, eval_ks)
        print("  Validation metrics:")
        for key, value in val_metrics.items():
            print(f"    {key}: {value:.6f}")

        # Also compute GT stats
        gt_sizes = [len(gt_map[bid]) for bid in val_bundle_ids if bid in gt_map]
        print(f"\n  GT avg products/bundle: {np.mean(gt_sizes):.1f} "
              f"(min={min(gt_sizes)}, max={max(gt_sizes)}, "
              f"median={np.median(gt_sizes):.0f})")

        metrics_out = output_dir / "val_metrics_top1.json"
        metrics_out.write_text(json.dumps(val_metrics, indent=2))
    else:
        print("\n[3/4] Validation skipped (val_ratio=0)")

    # ---- Test submission ----
    print(f"\n[4/4] Generating test submission...")
    test_bundle_ids = test_df["bundle_asset_id"].astype(str).drop_duplicates().tolist()
    print(f"  Test bundles: {len(test_bundle_ids)}")

    test_boxes = detect_bundles(test_bundle_ids, bundle_image_map, cfg.params)
    detection_stats = {
        "total": len(test_bundle_ids),
        "with_detections": sum(1 for boxes in test_boxes.values() if boxes),
        "avg_boxes": np.mean([len(b) for b in test_boxes.values()]) if test_boxes else 0,
    }
    print(f"  Detections: {detection_stats}")

    test_crop_items = build_crop_items_no_fallback(
        test_bundle_ids, bundle_image_map, test_boxes,
    )
    test_crop_embs = encode_bundle_crops(
        test_crop_items, model, preprocess, device, batch_size, num_workers, amp,
    )
    test_predictions = retrieve_top1_per_crop(
        test_crop_embs, search_index,
        bundle_to_gender=bundle_to_gender,
        product_to_gender=product_to_gender,
        gender_filter=gender_filter,
        score_threshold=score_threshold,
    )

    submission_rows: List[Dict[str, str]] = []
    for bid in test_bundle_ids:
        preds = test_predictions.get(bid, [])
        for pid in preds:
            submission_rows.append({"bundle_asset_id": bid, "product_asset_id": pid})

    submission_df = pd.DataFrame(submission_rows, columns=["bundle_asset_id", "product_asset_id"])
    submission_out = output_dir / "test_submission_top1.csv"
    submission_df.to_csv(submission_out, index=False)

    # Stats
    preds_per_bundle = [len(test_predictions.get(bid, [])) for bid in test_bundle_ids]
    print(f"\n{'=' * 60}")
    print(f"Submission saved: {submission_out} ({len(submission_df)} rows)")
    print(f"Bundles: {len(test_bundle_ids)}")
    print(f"Avg products/bundle: {np.mean(preds_per_bundle):.1f} "
          f"(min={min(preds_per_bundle)}, max={max(preds_per_bundle)}, "
          f"median={np.median(preds_per_bundle):.0f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
