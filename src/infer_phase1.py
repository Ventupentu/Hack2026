"""Phase 1 inference: section filtering + zero-shot crop classification + top-3/crop.

Three key improvements over infer_top1.py:
  1. Section-based product filtering  — separate product index per section
  2. Zero-shot crop classification    — CLIP classifies each crop's garment type
  3. Top-3 per crop + category boost  — more candidates, smarter ranking

Usage:
    python -m src.infer_phase1 \
        infer.checkpoint_path=outputs/2026-02-28/11-49-02/retrieval_openclip/best.pt \
        infer.val_ratio=0.1
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

import open_clip
import hydra
from hydra.utils import to_absolute_path

from src.config import InditexConfig
from src.new_infer import (
    ProductIndex,
    build_image_map,
    collate_skip_none,
    detect_bundles,
    encode_products,
    filter_cross_gender,
    load_bundle_genders,
    load_openclip_model,
    load_product_categories,
    load_product_genders,
    parse_ks,
    resolve_device,
    set_seed,
    BundleCropDataset,
    _encode_loader,
    open_image_safe,
    crop_with_box,
)
from src.utils.metrics import evaluate_bundle_retrieval

_GENDER_UNKNOWN = 0


# ---------------------------------------------------------------------------
# Section mapping: description -> allowed sections
# ---------------------------------------------------------------------------

def build_desc_to_sections(
    train_df: pd.DataFrame,
    products_df: pd.DataFrame,
    bundles_df: pd.DataFrame,
) -> Dict[str, Set[int]]:
    """From training data, learn which product descriptions appear in which sections."""
    merged = train_df.merge(
        products_df[["product_asset_id", "product_description"]],
        on="product_asset_id", how="left",
    ).merge(
        bundles_df[["bundle_asset_id", "bundle_id_section"]],
        on="bundle_asset_id", how="left",
    )

    desc_to_sections: Dict[str, Set[int]] = defaultdict(set)
    for _, row in merged.iterrows():
        desc = str(row["product_description"]).strip().upper()
        sec = row["bundle_id_section"]
        if pd.notna(sec):
            desc_to_sections[desc].add(int(sec))

    return dict(desc_to_sections)


def assign_product_sections(
    products_df: pd.DataFrame,
    desc_to_sections: Dict[str, Set[int]],
) -> Dict[str, Set[int]]:
    """Assign each product to allowed sections based on its description."""
    product_sections: Dict[str, Set[int]] = {}
    all_sections = {1, 2, 3}

    for _, row in products_df.iterrows():
        pid = str(row["product_asset_id"])
        desc = str(row["product_description"]).strip().upper()
        sections = desc_to_sections.get(desc, all_sections)  # unknown -> all
        product_sections[pid] = sections

    return product_sections


def build_section_indices(
    encoded_pids: List[str],
    product_embeddings: np.ndarray,
    product_sections: Dict[str, Set[int]],
) -> Dict[int, Tuple[ProductIndex, List[str]]]:
    """Build one ProductIndex per section, filtering embeddings accordingly."""
    section_indices: Dict[int, Tuple[ProductIndex, List[str]]] = {}

    for section in [1, 2, 3]:
        mask = [i for i, pid in enumerate(encoded_pids) if section in product_sections.get(pid, {1, 2, 3})]
        sec_pids = [encoded_pids[i] for i in mask]
        sec_embs = product_embeddings[mask]
        sec_index = ProductIndex(sec_pids, sec_embs)
        section_indices[section] = (sec_index, sec_pids)
        print(f"    Section {section}: {len(sec_pids)} products")

    return section_indices


# ---------------------------------------------------------------------------
# Zero-shot crop classification
# ---------------------------------------------------------------------------

@torch.inference_mode()
def build_category_text_embeddings(
    categories: List[str],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor]:
    """Pre-compute text embeddings for all product categories using CLIP."""
    prompts = [f"a photo of {cat.lower()}" for cat in categories]
    tokens = tokenizer(prompts).to(device)

    # Use the underlying CLIP model's text encoder
    clip_model = model.clip_model if hasattr(model, "clip_model") else model
    text_features = clip_model.encode_text(tokens)
    text_features = F.normalize(text_features.float(), p=2, dim=1)

    return categories, text_features


def classify_crop(
    crop_embedding: np.ndarray,
    category_names: List[str],
    category_embeddings: torch.Tensor,
    device: torch.device,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Classify a crop into the most likely garment categories."""
    query = torch.from_numpy(crop_embedding).float().unsqueeze(0).to(device)
    sims = (query @ category_embeddings.T).squeeze(0)
    topk_vals, topk_idxs = torch.topk(sims, min(top_k, len(category_names)))

    return [
        (category_names[idx.item()], val.item())
        for val, idx in zip(topk_vals, topk_idxs)
    ]


# ---------------------------------------------------------------------------
# Build crop items (no full-image fallback when detections exist)
# ---------------------------------------------------------------------------

def build_crop_items_clean(
    bundle_ids: List[str],
    bundle_image_map: Dict[str, Path],
    bundle_to_boxes: Dict[str, list],
) -> List[Tuple[str, Path, Any, float]]:
    """Build crop items WITHOUT full-image fallback when YOLO detections exist."""
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
        else:
            items.append((bid, path, None, 1.0))

    return items


# ---------------------------------------------------------------------------
# Encode bundle crops (no hflip TTA — we average original+hflip per crop)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_bundle_crops_averaged(
    crop_items: List[Tuple[str, Path, Any, float]],
    model: torch.nn.Module,
    preprocess: Callable,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    """Encode bundle crops WITH hflip TTA, averaging original+hflip per physical crop.

    Returns bundle_id -> [(averaged_embedding, confidence), ...] where each entry
    is one physical crop (not duplicated by TTA).
    """
    from torch.utils.data import DataLoader

    dataset = BundleCropDataset(crop_items, preprocess, use_hflip_tta=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    ids, embs, confs = _encode_loader(model, loader, device, amp, "Encoding bundle crops")

    # Group by (bundle_id, confidence) pairs to identify physical crops
    # The TTA produces original (False) + hflip (True) with same (bid, conf)
    # We average these pairs
    crop_groups: Dict[Tuple[str, float], List[np.ndarray]] = defaultdict(list)
    for bid, emb, conf in zip(ids, embs, confs):
        key = (bid, round(float(conf), 6))
        crop_groups[key].append(emb)

    # Average TTA embeddings per physical crop and re-normalize
    result: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
    for (bid, conf), emb_list in crop_groups.items():
        avg_emb = np.mean(emb_list, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
        result[bid].append((avg_emb.astype(np.float32), conf))

    return dict(result)


# ---------------------------------------------------------------------------
# Phase 1 retrieval: section filter + zero-shot category + top-3/crop
# ---------------------------------------------------------------------------

def retrieve_phase1(
    bundle_crops: Dict[str, List[Tuple[np.ndarray, float]]],
    section_indices: Dict[int, Tuple[ProductIndex, List[str]]],
    bundle_to_section: Dict[str, int],
    product_to_desc: Dict[str, str],
    category_names: List[str],
    category_embeddings: torch.Tensor,
    device: torch.device,
    top_k_per_crop: int = 3,
    category_boost: float = 1.5,
    max_products: int = 15,
    max_per_category: int = 2,
    bundle_to_gender: Optional[Dict[str, int]] = None,
    product_to_gender: Optional[Dict[str, int]] = None,
    gender_filter: bool = True,
) -> Dict[str, List[str]]:
    """Phase 1 retrieval pipeline:

    For each bundle:
      1. Get the section-filtered product index
      2. For each crop:
         a. Classify the crop (zero-shot) to get predicted category
         b. Search top-K from section index
         c. Boost products matching the predicted category
      3. Aggregate scores, apply filters, deduplicate by category
    """
    results: Dict[str, List[str]] = {}
    fallback_index = section_indices.get(1, list(section_indices.values())[0])

    for bundle_id, crops in tqdm(bundle_crops.items(), desc="Phase 1 retrieval", leave=False):
        section = bundle_to_section.get(bundle_id, 0)
        index_entry = section_indices.get(section, fallback_index)
        product_index, _ = index_entry

        candidates: Dict[str, float] = {}

        for emb, conf in crops:
            # Step 1: Classify the crop (zero-shot)
            predicted_cats = classify_crop(
                emb, category_names, category_embeddings, device, top_k=3,
            )
            top_category = predicted_cats[0][0] if predicted_cats else ""

            # Step 2: Search top-K from section-filtered index
            indices, scores = product_index.search(emb, top_k_per_crop)

            for idx, sim in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                pid = product_index.product_ids[int(idx)]
                combined = float(sim) * max(float(conf), 0.1)

                # Step 3: Category boost — if product description matches predicted category
                prod_desc = product_to_desc.get(pid, "")
                if top_category and prod_desc == top_category:
                    combined *= category_boost

                # MAX aggregation: keep the best score for each product
                candidates[pid] = max(candidates.get(pid, 0.0), combined)

        # Sort by score descending
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        # Gender filter
        if gender_filter and bundle_to_gender and product_to_gender:
            bundle_gender = bundle_to_gender.get(bundle_id, _GENDER_UNKNOWN)
            ranked = filter_cross_gender(ranked, bundle_gender, product_to_gender)

        # Category Deduplication
        final_pids = []
        category_counts: Dict[str, int] = defaultdict(int)
        
        for pid, score in ranked:
            if len(final_pids) >= max_products:
                break
                
            # Category limit logic
            desc = product_to_desc.get(pid, "")
            if desc and category_counts[desc] >= max_per_category:
                continue
                
            final_pids.append(pid)
            if desc:
                category_counts[desc] += 1
                
        # If we couldn't find enough products meeting the strict category rules, 
        # we still want to return max_products if possible to maximize recall.
        # We fill the remaining slots with the next best overall candidates regardless of category limit, 
        # still respecting gender.
        if len(final_pids) < max_products:
            for pid, score in ranked:
                if len(final_pids) >= max_products:
                    break
                if pid not in final_pids:
                    final_pids.append(pid)

        results[bundle_id] = final_pids

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    bundle_ids: List[str],
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    ks: List[int],
) -> Dict[str, float]:
    pred_ids, pred_lists = [], []
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
    gender_filter = bool(getattr(cfg.infer, "gender_filter", True))

    # Phase 1 specific params
    TOP_K_PER_CROP = 10      # top-K candidates per crop
    CATEGORY_BOOST = 1.8     # boost factor for category-matching products
    MAX_PRODUCTS = 15        # max products per bundle in output
    MAX_PER_CATEGORY = 2     # max products with the same description

    set_seed(seed)

    # ---- Load model ----
    print("=" * 60)
    print("INFER PHASE 1 — Section filter + Zero-shot + Top-K/crop")
    print("=" * 60)
    model, preprocess, tokenizer = load_openclip_model(checkpoint_path, device)
    print(f"Device: {device} | AMP: {amp} | TTA: {n_tta}")
    print(f"Top-K/crop: {TOP_K_PER_CROP} | Category boost: {CATEGORY_BOOST} | Max products: {MAX_PRODUCTS}")

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
    # Uppercase description map for category matching
    product_to_desc_upper: Dict[str, str] = {
        pid: desc.strip().upper() for pid, desc in product_to_text.items()
    }
    print(f"Products: {len(product_ids)} with images / {len(all_product_ids)} total")

    # ---- Build section mapping ----
    print("\n[0/5] Building section mapping...")
    desc_to_sections = build_desc_to_sections(train_df, products_df, bundles_df)
    product_sections = assign_product_sections(products_df, desc_to_sections)

    bundle_to_section: Dict[str, int] = {}
    for _, row in bundles_df.iterrows():
        bid = str(row["bundle_asset_id"])
        sec = row["bundle_id_section"]
        if pd.notna(sec):
            bundle_to_section[bid] = int(sec)

    # ---- Load gender maps and override descriptions if available map ----
    bundle_to_gender = load_bundle_genders(bundles_df)
    products_gender_csv = data_dir / "product_dataset_with_gender.csv"
    product_to_gender = load_product_genders(products_gender_csv)
    
    if products_gender_csv.exists():
        pg_df = pd.read_csv(products_gender_csv)
        if "product_description" in pg_df.columns:
            for _, row in pg_df.iterrows():
                pid = str(row["product_asset_id"])
                desc = str(row["product_description"]).strip().upper()
                if pid in product_to_desc_upper and pd.notna(row["product_description"]) and desc:
                    product_to_desc_upper[pid] = desc

    # ---- Build zero-shot category embeddings ----
    print("\n[1/5] Building zero-shot category embeddings...")
    all_descriptions = sorted(set(
        desc.strip().upper()
        for desc in products_df["product_description"].dropna().unique()
        if desc.strip()
    ))
    category_names, category_embeddings = build_category_text_embeddings(
        all_descriptions, model, tokenizer, device,
    )
    print(f"  Categories: {len(category_names)}")

    # ---- Encode products ----
    print(f"\n[2/5] Encoding products (TTA={n_tta})...")
    encoded_pids, product_embeddings = encode_products(
        product_ids, product_image_map, product_to_text,
        model, preprocess, tokenizer, device,
        batch_size, num_workers, amp, n_tta,
    )
    print(f"  Product embeddings: {product_embeddings.shape}")

    # ---- Build per-section indices ----
    print("\n[3/5] Building per-section product indices...")
    section_indices = build_section_indices(
        encoded_pids, product_embeddings, product_sections,
    )

    # ---- Validation ----
    if val_ratio > 0:
        print(f"\n[4/5] Validation (val_ratio={val_ratio})...")
        train_bundle_ids = train_df["bundle_asset_id"].astype(str).tolist()
        unique_train_bundles = list(dict.fromkeys(train_bundle_ids))
        random.Random(seed).shuffle(unique_train_bundles)
        val_size = max(1, int(round(len(unique_train_bundles) * val_ratio)))
        val_bundle_ids = sorted(unique_train_bundles[:val_size])
        print(f"  Val bundles: {len(val_bundle_ids)}")

        gt_map: Dict[str, Set[str]] = defaultdict(set)
        for row in train_df.itertuples(index=False):
            gt_map[str(row.bundle_asset_id)].add(str(row.product_asset_id))

        val_boxes = detect_bundles(val_bundle_ids, bundle_image_map, cfg.params)
        val_crop_items = build_crop_items_clean(val_bundle_ids, bundle_image_map, val_boxes)
        val_crop_embs = encode_bundle_crops_averaged(
            val_crop_items, model, preprocess, device, batch_size, num_workers, amp,
        )
        val_predictions = retrieve_phase1(
            val_crop_embs, section_indices, bundle_to_section,
            product_to_desc_upper, category_names, category_embeddings, device,
            top_k_per_crop=TOP_K_PER_CROP,
            category_boost=CATEGORY_BOOST,
            max_products=MAX_PRODUCTS,
            max_per_category=MAX_PER_CATEGORY,
            bundle_to_gender=bundle_to_gender,
            product_to_gender=product_to_gender,
            gender_filter=gender_filter,
        )

        num_preds = [len(v) for v in val_predictions.values()]
        print(f"  Avg products/bundle: {np.mean(num_preds):.1f} "
              f"(min={min(num_preds)}, max={max(num_preds)}, "
              f"median={np.median(num_preds):.0f})")

        val_metrics = validate(val_bundle_ids, val_predictions, gt_map, eval_ks)
        print("  Validation metrics:")
        for key, value in val_metrics.items():
            print(f"    {key}: {value:.6f}")

        gt_sizes = [len(gt_map[bid]) for bid in val_bundle_ids if bid in gt_map]
        print(f"\n  GT avg products/bundle: {np.mean(gt_sizes):.1f}")

        metrics_out = output_dir / "val_metrics_phase1.json"
        metrics_out.write_text(json.dumps(val_metrics, indent=2))
    else:
        print("\n[4/5] Validation skipped")

    # ---- Test submission ----
    print(f"\n[5/5] Generating test submission...")
    test_bundle_ids = test_df["bundle_asset_id"].astype(str).drop_duplicates().tolist()
    print(f"  Test bundles: {len(test_bundle_ids)}")

    test_boxes = detect_bundles(test_bundle_ids, bundle_image_map, cfg.params)
    detection_stats = {
        "total": len(test_bundle_ids),
        "with_detections": sum(1 for boxes in test_boxes.values() if boxes),
        "avg_boxes": float(np.mean([len(b) for b in test_boxes.values()])) if test_boxes else 0,
    }
    print(f"  Detections: {detection_stats}")

    test_crop_items = build_crop_items_clean(test_bundle_ids, bundle_image_map, test_boxes)
    test_crop_embs = encode_bundle_crops_averaged(
        test_crop_items, model, preprocess, device, batch_size, num_workers, amp,
    )
    test_predictions = retrieve_phase1(
        test_crop_embs, section_indices, bundle_to_section,
        product_to_desc_upper, category_names, category_embeddings, device,
        top_k_per_crop=TOP_K_PER_CROP,
        category_boost=CATEGORY_BOOST,
        max_products=MAX_PRODUCTS,
        max_per_category=MAX_PER_CATEGORY,
        bundle_to_gender=bundle_to_gender,
        product_to_gender=product_to_gender,
        gender_filter=gender_filter,
    )

    submission_rows: List[Dict[str, str]] = []
    
    # We NEED exactly 15 rows per bundle, no more, no less, for the leaderboard evaluater
    fallback_pids = list(products_df["product_asset_id"].unique())
    
    for bid in test_bundle_ids:
        preds = test_predictions.get(bid, [])
        used_pids = set(preds)
        
        # Add predictions we actually got
        for pid in preds[:MAX_PRODUCTS]:
            submission_rows.append({"bundle_asset_id": bid, "product_asset_id": pid})
            
        # Pad with fallbacks if we have less than 15
        needed = MAX_PRODUCTS - min(len(preds), MAX_PRODUCTS)
        if needed > 0:
            for fallback in fallback_pids:
                if fallback not in used_pids:
                    submission_rows.append({"bundle_asset_id": bid, "product_asset_id": fallback})
                    used_pids.add(fallback)
                    needed -= 1
                    if needed == 0:
                        break

    submission_df = pd.DataFrame(submission_rows, columns=["bundle_asset_id", "product_asset_id"])
    submission_out = output_dir / "test_submission_phase1.csv"
    submission_df.to_csv(submission_out, index=False)

    print(f"\n{'=' * 60}")
    print(f"Submission saved: {submission_out} ({len(submission_df)} rows)")
    print(f"Bundles: {len(test_bundle_ids)}")
    print(f"Items per bundle: EXACTLY {MAX_PRODUCTS} (padded if needed)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
