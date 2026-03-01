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
from src.reranker import apply_hubness_penalty, heavy_model_rerank

_GENDER_UNKNOWN = 0


# ---------------------------------------------------------------------------
# Lightweight MLP reranker (trained on embedding pair features)
# ---------------------------------------------------------------------------

class MLPReranker(torch.nn.Module):
    """Small MLP scorer for (query_emb, product_emb) pairs."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float) -> None:
        super().__init__()
        layers: List[torch.nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _build_pair_features(
    query_vecs: torch.Tensor,
    product_vecs: torch.Tensor,
    feature_cfg: Dict[str, bool],
) -> torch.Tensor:
    """Build pairwise features: sim, |q-p|, q*p (+ optional extras)."""
    sim = (query_vecs * product_vecs).sum(dim=1, keepdim=True)
    parts = [sim]

    if feature_cfg.get("use_abs_diff", True):
        parts.append(torch.abs(query_vecs - product_vecs))
    if feature_cfg.get("use_elem_product", True):
        parts.append(query_vecs * product_vecs)
    if feature_cfg.get("use_sq_diff", False):
        diff = query_vecs - product_vecs
        parts.append(diff * diff)
    if feature_cfg.get("use_raw_concat", False):
        parts.extend([query_vecs, product_vecs])

    return torch.cat(parts, dim=1)


def _aggregate_bundle_embedding(crops: List[Tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
    """Aggregate crop embeddings into one query embedding per bundle."""
    if not crops:
        return None
    weighted = np.zeros_like(crops[0][0], dtype=np.float32)
    weight_sum = 0.0
    for emb, conf in crops:
        w = max(float(conf), 0.1)
        weighted += (emb.astype(np.float32) * w)
        weight_sum += w
    if weight_sum <= 0:
        return None
    out = weighted / weight_sum
    norm = float(np.linalg.norm(out))
    if norm <= 1e-8:
        return None
    return (out / norm).astype(np.float32)


def aggregate_bundle_embedding_map(
    bundle_crop_embeddings: Dict[str, List[Tuple[np.ndarray, float]]],
    bundle_ids: Optional[List[str]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Aggregate multi-crop bundle embeddings into one vector per bundle.

    Uses confidence-weighted mean over crop embeddings, then L2 normalization.
    Returns:
      - ordered bundle_ids
      - embeddings matrix [N, D]
      - num_crops per bundle [N]
    """
    if bundle_ids is None:
        bundle_ids = sorted(bundle_crop_embeddings.keys())

    out_ids: List[str] = []
    out_embs: List[np.ndarray] = []
    out_num_crops: List[int] = []
    for bid in bundle_ids:
        crops = bundle_crop_embeddings.get(bid, [])
        agg = _aggregate_bundle_embedding(crops)
        if agg is None:
            continue
        out_ids.append(bid)
        out_embs.append(agg.astype(np.float32))
        out_num_crops.append(len(crops))

    if not out_embs:
        return [], np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return (
        out_ids,
        np.stack(out_embs).astype(np.float32),
        np.asarray(out_num_crops, dtype=np.int32),
    )


def load_mlp_reranker(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[MLPReranker, Dict[str, bool], float, int]:
    """Load trained MLP reranker checkpoint produced by train_mlp.py."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MLP reranker checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(payload)!r}")

    model_state = payload.get("model_state")
    input_dim = int(payload.get("input_dim", 0))
    hidden_dims = [int(v) for v in payload.get("hidden_dims", [512, 128])]
    dropout = float(payload.get("dropout", 0.2))
    embedding_dim = int(payload.get("embedding_dim", 0))
    feature_cfg_raw = payload.get("feature_config", {})
    if model_state is None or input_dim <= 0:
        raise ValueError(
            "Invalid MLP checkpoint: expected keys `model_state` and positive `input_dim`."
        )

    feature_cfg = {
        "use_abs_diff": bool(feature_cfg_raw.get("use_abs_diff", True)),
        "use_elem_product": bool(feature_cfg_raw.get("use_elem_product", True)),
        "use_sq_diff": bool(feature_cfg_raw.get("use_sq_diff", False)),
        "use_raw_concat": bool(feature_cfg_raw.get("use_raw_concat", False)),
    }
    blend_alpha = float(payload.get("blend_alpha", 0.2))

    model = MLPReranker(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, feature_cfg, blend_alpha, embedding_dim


@torch.inference_mode()
def apply_mlp_rerank(
    predictions: Dict[str, List[Tuple[str, float]]],
    bundle_crop_embeddings: Dict[str, List[Tuple[np.ndarray, float]]],
    encoded_product_ids: List[str],
    product_embeddings: np.ndarray,
    model: MLPReranker,
    feature_cfg: Dict[str, bool],
    blend_alpha: float,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, List[Tuple[str, float]]]:
    """Rerank candidates using trained MLP over embedding pair features."""
    if not predictions:
        return predictions

    pid_to_index = {pid: i for i, pid in enumerate(encoded_product_ids)}
    product_tensor = torch.from_numpy(product_embeddings.astype(np.float32)).to(device)

    out: Dict[str, List[Tuple[str, float]]] = {}
    for bundle_id, preds in tqdm(predictions.items(), desc="MLP Rerank", leave=False):
        if not preds:
            out[bundle_id] = preds
            continue

        q_emb = _aggregate_bundle_embedding(bundle_crop_embeddings.get(bundle_id, []))
        if q_emb is None:
            out[bundle_id] = preds
            continue

        # Normalize base scores per bundle for a stable blend with MLP logits.
        base_scores = np.asarray([float(score) for _, score in preds], dtype=np.float32)
        base_mean = float(base_scores.mean())
        base_std = float(base_scores.std())
        if base_std > 1e-6:
            base_scaled = (base_scores - base_mean) / base_std
        else:
            base_scaled = np.zeros_like(base_scores)

        mlp_scores = np.full(len(preds), fill_value=-1e9, dtype=np.float32)
        valid_rows: List[int] = []
        valid_prod_idx: List[int] = []
        for row_idx, (pid, _) in enumerate(preds):
            idx = pid_to_index.get(pid)
            if idx is not None:
                valid_rows.append(row_idx)
                valid_prod_idx.append(idx)

        if valid_rows:
            q_tensor = torch.from_numpy(q_emb).to(device).unsqueeze(0)
            prod_idx_tensor = torch.tensor(valid_prod_idx, dtype=torch.long, device=device)
            prod_vecs = product_tensor.index_select(0, prod_idx_tensor)

            # Score in chunks to control memory for larger candidate pools.
            scores_chunks: List[np.ndarray] = []
            for start in range(0, len(valid_rows), batch_size):
                end = min(start + batch_size, len(valid_rows))
                q_chunk = q_tensor.expand(end - start, -1)
                p_chunk = prod_vecs[start:end]
                feats = _build_pair_features(q_chunk, p_chunk, feature_cfg)
                s = model(feats).detach().cpu().numpy().astype(np.float32)
                scores_chunks.append(s)

            flat_scores = np.concatenate(scores_chunks, axis=0) if scores_chunks else np.empty((0,), dtype=np.float32)
            for row_idx, s in zip(valid_rows, flat_scores.tolist()):
                mlp_scores[row_idx] = float(s)

        final_preds: List[Tuple[str, float]] = []
        for i, (pid, _) in enumerate(preds):
            final_score = float(mlp_scores[i] + (blend_alpha * base_scaled[i]))
            final_preds.append((pid, final_score))
        out[bundle_id] = sorted(final_preds, key=lambda x: x[1], reverse=True)

    return out


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
    bundle_to_article: Optional[Dict[str, str]] = None,
    product_to_article: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Phase 1 retrieval pipeline:

    For each bundle:
      1. Get the section-filtered product index
      2. For each crop:
         a. Classify the crop (zero-shot) to get predicted category
         b. Search top-K from section index
         c. Boost products matching the predicted category
      3. Aggregate scores, apply filters, deduplicate by category
    """
    results: Dict[str, List[Tuple[str, float]]] = {}
    fallback_index = section_indices.get(1, list(section_indices.values())[0])

    for bundle_id, crops in tqdm(bundle_crops.items(), desc="Phase 1 retrieval", leave=False):
        section = bundle_to_section.get(bundle_id, 0)
        index_entry = section_indices.get(section, fallback_index)
        product_index, _ = index_entry

        candidates: Dict[str, float] = {}
        bundle_article = bundle_to_article.get(bundle_id) if bundle_to_article else None

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
                    
                # Massive Boost for Exact Article Match (Golden Feature)
                if bundle_article and product_to_article:
                    prod_article = product_to_article.get(pid)
                    if prod_article and bundle_article == prod_article:
                        combined *= 1000.0

                # MAX aggregation: keep the best score for each product
                candidates[pid] = max(candidates.get(pid, 0.0), combined)

        # Sort by score descending
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        # Gender filter
        if gender_filter and bundle_to_gender and product_to_gender:
            bundle_gender = bundle_to_gender.get(bundle_id, _GENDER_UNKNOWN)
            ranked = filter_cross_gender(ranked, bundle_gender, product_to_gender)

        # Category Deduplication
        final_preds = []
        category_counts: Dict[str, int] = defaultdict(int)
        
        for pid, score in ranked:
            if len(final_preds) >= max_products:
                break
                
            # Category limit logic
            desc = product_to_desc.get(pid, "")
            if desc and category_counts[desc] >= max_per_category:
                continue
                
            final_preds.append((pid, score))
            if desc:
                category_counts[desc] += 1
                
        # If we couldn't find enough products meeting the strict category rules, 
        # we still want to return max_products if possible to maximize recall.
        # We fill the remaining slots with the next best overall candidates regardless of category limit, 
        # still respecting gender.
        if len(final_preds) < max_products:
            added_pids = {p for p, _ in final_preds}
            for pid, score in ranked:
                if len(final_preds) >= max_products:
                    break
                if pid not in added_pids:
                    final_preds.append((pid, score))
                    added_pids.add(pid)

        results[bundle_id] = final_preds

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    bundle_ids: List[str],
    predictions: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    ks: List[int],
) -> Dict[str, float]:
    pred_ids, pred_lists = [], []
    for bid in bundle_ids:
        if bid in predictions and bid in ground_truth:
            pred_ids.append(bid)
            pred_lists.append([p for p, _ in predictions[bid]])
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
    
    rerank_hubness_enabled = bool(getattr(cfg.infer, "rerank_hubness_enabled", False))
    rerank_hubness_max_ratio = float(getattr(cfg.infer, "rerank_hubness_max_ratio", 0.01))
    rerank_hubness_penalty = float(getattr(cfg.infer, "rerank_hubness_penalty", 0.1))
    rerank_heavy_enabled = bool(getattr(cfg.infer, "rerank_heavy_enabled", False))
    rerank_heavy_model = str(getattr(cfg.infer, "rerank_heavy_model", "ViT-SO400M-14-SigLIP-384"))
    rerank_heavy_pretrained = str(getattr(cfg.infer, "rerank_heavy_pretrained", "webli"))
    rerank_heavy_weight = float(getattr(cfg.infer, "rerank_heavy_weight", 0.4))
    rerank_mlp_enabled = bool(getattr(cfg.infer, "rerank_mlp_enabled", False))
    rerank_mlp_checkpoint = str(getattr(cfg.infer, "rerank_mlp_checkpoint", "")).strip()
    rerank_mlp_blend_alpha_cfg = float(getattr(cfg.infer, "rerank_mlp_blend_alpha", -1.0))
    rerank_mlp_batch_size = int(getattr(cfg.infer, "rerank_mlp_batch_size", 4096))
    rerank_mlp_candidate_pool = int(getattr(cfg.infer, "rerank_mlp_candidate_pool", 200))
    export_train_bundle_embeddings = bool(getattr(cfg.infer, "export_train_bundle_embeddings", False))
    train_bundle_embeddings_out = str(
        getattr(cfg.infer, "train_bundle_embeddings_out", "outputs/train_bundle_embeddings.pt")
    ).strip()

    # Phase 1 specific params
    TOP_K_PER_CROP = 20     # top-K candidates per crop
    CATEGORY_BOOST = 1.4     # boost factor for category-matching products
    MAX_PRODUCTS = 15        # max products per bundle in output
    MAX_PER_CATEGORY = 2     # max products with the same description
    RETRIEVAL_POOL = max(MAX_PRODUCTS * 4, rerank_mlp_candidate_pool if rerank_mlp_enabled else 0)

    set_seed(seed)

    # ---- Load model ----
    print("=" * 60)
    print("INFER PHASE 1 — Section filter + Zero-shot + Top-K/crop")
    print("=" * 60)
    model, preprocess, tokenizer = load_openclip_model(checkpoint_path, device)
    print(f"Device: {device} | AMP: {amp} | TTA: {n_tta}")
    print(
        f"Top-K/crop: {TOP_K_PER_CROP} | Category boost: {CATEGORY_BOOST} | "
        f"Retrieval pool: {RETRIEVAL_POOL} | Max products: {MAX_PRODUCTS}"
    )

    # ---- Load data ----
    bundles_df = pd.read_csv(bundles_csv)
    products_df = pd.read_csv(products_csv)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    bundle_image_map = build_image_map(bundle_images_dir)
    product_image_map = build_image_map(product_images_dir)
    
    def extract_article(url):
        if pd.isna(url):
            return None
        parts = str(url).split('/')
        for p in reversed(parts):
            if '.jpg' in p:
                return p.split('.jpg')[0].split('-')[0]
        return None

    bundles_df['article'] = bundles_df['bundle_image_url'].apply(extract_article)
    products_df['article'] = products_df['product_image_url'].apply(extract_article)
    
    bundle_to_article = dict(zip(bundles_df['bundle_asset_id'].astype(str), bundles_df['article']))
    product_to_article = dict(zip(products_df['product_asset_id'].astype(str), products_df['article']))

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
    
    embeds_out = output_dir / "product_embeddings.pt"
    print(f"  Saving product embeddings to {embeds_out}...")
    torch.save({
        "pids": encoded_pids,
        "embeddings": torch.tensor(product_embeddings)
    }, embeds_out)

    if export_train_bundle_embeddings:
        print("\n[2c/5] Exporting train bundle embeddings (multi-box aggregated)...")
        train_bundle_ids_for_export = (
            train_df["bundle_asset_id"].astype(str).drop_duplicates().tolist()
        )
        print(f"  Train bundles to export: {len(train_bundle_ids_for_export)}")

        train_boxes = detect_bundles(train_bundle_ids_for_export, bundle_image_map, cfg.params)
        train_crop_items = build_crop_items_clean(
            train_bundle_ids_for_export, bundle_image_map, train_boxes
        )
        train_crop_embs = encode_bundle_crops_averaged(
            train_crop_items, model, preprocess, device, batch_size, num_workers, amp,
        )
        agg_bundle_ids, agg_bundle_embs, agg_num_crops = aggregate_bundle_embedding_map(
            train_crop_embs, bundle_ids=train_bundle_ids_for_export
        )
        if agg_bundle_embs.size == 0:
            raise RuntimeError("No aggregated train bundle embeddings could be exported.")

        out_path = Path(to_absolute_path(train_bundle_embeddings_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bundle_ids": agg_bundle_ids,
                "embeddings": torch.tensor(agg_bundle_embs),
                "num_crops": torch.tensor(agg_num_crops, dtype=torch.int32),
                "aggregation": "confidence_weighted_mean_l2norm",
            },
            out_path,
        )
        print(
            f"  Saved train bundle embeddings: {out_path} "
            f"(shape={agg_bundle_embs.shape}, avg_crops={float(np.mean(agg_num_crops)):.2f})"
        )

    mlp_reranker: Optional[MLPReranker] = None
    mlp_feature_cfg: Dict[str, bool] = {}
    mlp_blend_alpha = rerank_mlp_blend_alpha_cfg
    if rerank_mlp_enabled:
        if not rerank_mlp_checkpoint:
            raise ValueError("infer.rerank_mlp_enabled=true requires infer.rerank_mlp_checkpoint")
        mlp_checkpoint_path = Path(to_absolute_path(rerank_mlp_checkpoint))
        print(f"\n[2b/5] Loading MLP reranker from {mlp_checkpoint_path}...")
        mlp_reranker, mlp_feature_cfg, ckpt_blend_alpha, mlp_emb_dim = load_mlp_reranker(
            checkpoint_path=mlp_checkpoint_path,
            device=device,
        )
        if mlp_emb_dim > 0 and mlp_emb_dim != int(product_embeddings.shape[1]):
            raise ValueError(
                f"MLP embedding_dim mismatch: checkpoint={mlp_emb_dim} "
                f"vs current={int(product_embeddings.shape[1])}"
            )
        if mlp_blend_alpha < 0:
            mlp_blend_alpha = ckpt_blend_alpha
        print(
            f"  MLP rerank enabled | blend_alpha={mlp_blend_alpha:.4f} "
            f"| batch_size={rerank_mlp_batch_size}"
        )

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
            max_products=RETRIEVAL_POOL,
            max_per_category=MAX_PER_CATEGORY,
            bundle_to_gender=bundle_to_gender,
            product_to_gender=product_to_gender,
            gender_filter=gender_filter,
            bundle_to_article=bundle_to_article,
            product_to_article=product_to_article,
        )

        if rerank_mlp_enabled and mlp_reranker is not None:
            print("  Applying MLP Reranker (Val)...")
            val_predictions = apply_mlp_rerank(
                predictions=val_predictions,
                bundle_crop_embeddings=val_crop_embs,
                encoded_product_ids=encoded_pids,
                product_embeddings=product_embeddings,
                model=mlp_reranker,
                feature_cfg=mlp_feature_cfg,
                blend_alpha=mlp_blend_alpha,
                device=device,
                batch_size=rerank_mlp_batch_size,
            )

        if rerank_hubness_enabled:
            print("  Applying Hubness Penalty (Val)...")
            val_predictions = apply_hubness_penalty(
                val_predictions, rerank_hubness_max_ratio, rerank_hubness_penalty
            )
            
        if rerank_heavy_enabled:
            val_predictions = heavy_model_rerank(
                predictions=val_predictions,
                bundle_crops=val_crop_items,
                products_df=products_df,
                product_images_dir=str(product_images_dir),
                device=device,
                model_name=rerank_heavy_model,
                pretrained=rerank_heavy_pretrained,
                heavy_weight=rerank_heavy_weight
            )

        # Truncate to MAX_PRODUCTS before validation
        val_predictions = {bid: preds[:MAX_PRODUCTS] for bid, preds in val_predictions.items()}

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
        max_products=RETRIEVAL_POOL,
        max_per_category=MAX_PER_CATEGORY,
        bundle_to_gender=bundle_to_gender,
        product_to_gender=product_to_gender,
        gender_filter=gender_filter,
        bundle_to_article=bundle_to_article,
        product_to_article=product_to_article,
    )

    if rerank_mlp_enabled and mlp_reranker is not None:
        print("  Applying MLP Reranker (Test)...")
        test_predictions = apply_mlp_rerank(
            predictions=test_predictions,
            bundle_crop_embeddings=test_crop_embs,
            encoded_product_ids=encoded_pids,
            product_embeddings=product_embeddings,
            model=mlp_reranker,
            feature_cfg=mlp_feature_cfg,
            blend_alpha=mlp_blend_alpha,
            device=device,
            batch_size=rerank_mlp_batch_size,
        )

    if rerank_hubness_enabled:
        print("  Applying Hubness Penalty (Test)...")
        test_predictions = apply_hubness_penalty(
            test_predictions, rerank_hubness_max_ratio, rerank_hubness_penalty
        )
        
    if rerank_heavy_enabled:
        test_predictions = heavy_model_rerank(
            predictions=test_predictions,
            bundle_crops=test_crop_items,
            products_df=products_df,
            product_images_dir=str(product_images_dir),
            device=device,
            model_name=rerank_heavy_model,
            pretrained=rerank_heavy_pretrained,
            heavy_weight=rerank_heavy_weight
        )
        
    # Truncate to MAX_PRODUCTS for test submission
    test_predictions = {bid: preds[:MAX_PRODUCTS] for bid, preds in test_predictions.items()}

    submission_rows: List[Dict[str, str]] = []
    
    # We NEED exactly 15 rows per bundle, no more, no less, for the leaderboard evaluater
    fallback_pids = list(products_df["product_asset_id"].unique())
    
    for bid in test_bundle_ids:
        preds = test_predictions.get(bid, [])
        used_pids = set([p for p, _ in preds])
        
        # Add predictions we actually got
        for pid, _ in preds[:MAX_PRODUCTS]:
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
