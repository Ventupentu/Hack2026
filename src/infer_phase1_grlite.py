"""Phase 1 inference with GR-Lite fine-tuned model.

Adapts infer_phase1.py to use the GR-Lite retrieval checkpoint instead of
OpenCLIP.  Since GR-Lite is vision-only there is no zero-shot text
classification; category boosting uses simple description matching instead.

Usage:
    python -m src.infer_phase1_grlite \
        infer.checkpoint_path=outputs/2026-02-28/23-09-53/retrieval_gr_lite/best.pt \
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import hydra
from hydra.utils import to_absolute_path

from src.config import InditexConfig
from src.models.retrieval_grlite import (
    GRLiteTensorEncoder,
    ProjectionHead,
    _extract_features_from_outputs,
    load_grlite_base_model,
    torch_load_any,
)
from src.new_infer import (
    ProductIndex,
    build_image_map,
    collate_skip_none,
    detect_bundles,
    filter_cross_gender,
    load_bundle_genders,
    load_product_genders,
    parse_ks,
    resolve_device,
    set_seed,
    open_image_safe,
    crop_with_box,
)
from src.detection import BoxXYXY
from src.utils.metrics import evaluate_bundle_retrieval

_GENDER_UNKNOWN = 0


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_grlite_inference_model(
    checkpoint_path: str,
    device: torch.device,
    fallback_model_name: str = "srpone/gr-lite",
    fallback_input_size: int = 518,
) -> Tuple[torch.nn.Module, int]:
    """Load GR-Lite encoder from a training checkpoint.

    Returns (model, input_size).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch_load_any(ckpt_path, map_location="cpu")
    model_name = str(ckpt.get("model_name", fallback_model_name))
    input_size = int(ckpt.get("input_size", fallback_input_size))
    epoch = ckpt.get("epoch", "?")

    print(f"Loading GR-Lite base from: {model_name} (input_size={input_size})")
    base_model = load_grlite_base_model(model_name, device=device, input_size=input_size)

    # Detect proj_head keys to reconstruct ProjectionHead
    sd = ckpt["model"]
    proj_keys = [k for k in sd if k.startswith("proj_head.")]
    proj_head: Optional[ProjectionHead] = None
    if proj_keys:
        # Infer dims from weight shapes
        w_in = sd["proj_head.net.0.weight"]   # Linear(input, hidden)
        w_out = sd["proj_head.net.3.weight"]  # Linear(hidden, output)
        input_dim, hidden_dim, output_dim = w_in.shape[1], w_in.shape[0], w_out.shape[0]
        proj_head = ProjectionHead(input_dim, hidden_dim, output_dim)
        print(f"  ProjectionHead: {input_dim}→{hidden_dim}→{output_dim}")

    encoder = GRLiteTensorEncoder(base_model, proj_head=proj_head)
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")

    encoder = encoder.to(device).eval()
    print(f"Loaded GR-Lite checkpoint: {ckpt_path} (epoch={epoch})")
    return encoder, input_size


def build_preprocess(input_size: int) -> Callable:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------

class ImageDataset(Dataset):
    """Simple image dataset returning (id, tensor)."""

    def __init__(self, ids: List[str], id_to_path: Dict[str, Path], preprocess: Callable) -> None:
        self.ids = ids
        self.id_to_path = id_to_path
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        aid = self.ids[idx]
        img = open_image_safe(self.id_to_path[aid])
        if img is None:
            return None
        return {"id": aid, "img": self.preprocess(img)}


class CropDataset(Dataset):
    """Bundle crops with optional hflip TTA."""

    def __init__(
        self,
        crop_items: List[Tuple[str, Path, Any, float]],
        preprocess: Callable,
        use_hflip: bool = True,
    ) -> None:
        self.preprocess = preprocess
        self.items: List[Tuple[str, Path, Any, float, bool]] = []
        for bid, path, box, conf in crop_items:
            self.items.append((bid, path, box, conf, False))
            if use_hflip:
                self.items.append((bid, path, box, conf, True))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        bid, path, box, conf, do_flip = self.items[idx]
        img = open_image_safe(path)
        if img is None:
            return None
        if box is not None:
            img = crop_with_box(img, box)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return {"bundle_id": bid, "confidence": conf, "img": self.preprocess(img)}


# ------------------------------------------------------------------
# Encoding
# ------------------------------------------------------------------

@torch.inference_mode()
def encode_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    desc: str = "Encoding",
) -> Tuple[List[str], np.ndarray, List[float]]:
    all_ids: List[str] = []
    all_embs: List[np.ndarray] = []
    all_confs: List[float] = []

    for batch in tqdm(loader, leave=False, desc=desc):
        if batch is None:
            continue
        imgs = batch["img"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            feats = model(imgs)
        feats = F.normalize(feats.float(), p=2, dim=1)
        all_embs.append(feats.cpu().numpy())
        all_ids.extend(batch.get("id", batch.get("bundle_id")))
        if "confidence" in batch:
            all_confs.extend([float(c) for c in batch["confidence"]])

    if not all_embs:
        return [], np.zeros((0, 0), dtype=np.float32), []
    return all_ids, np.concatenate(all_embs, axis=0), all_confs


def encode_products(
    product_ids: List[str],
    product_image_map: Dict[str, Path],
    model: torch.nn.Module,
    preprocess: Callable,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> Tuple[List[str], np.ndarray]:
    ds = ImageDataset(product_ids, product_image_map, preprocess)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    ids, embs, _ = encode_loader(model, loader, device, amp, "Encoding products")
    return ids, embs


def encode_bundle_crops(
    crop_items: List[Tuple[str, Path, Any, float]],
    model: torch.nn.Module,
    preprocess: Callable,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    ds = CropDataset(crop_items, preprocess, use_hflip=True)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    ids, embs, confs = encode_loader(model, loader, device, amp, "Encoding bundle crops")

    # Average hflip TTA pairs per physical crop
    crop_groups: Dict[Tuple[str, float], List[np.ndarray]] = defaultdict(list)
    for bid, emb, conf in zip(ids, embs, confs):
        key = (bid, round(float(conf), 6))
        crop_groups[key].append(emb)

    result: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
    for (bid, conf), emb_list in crop_groups.items():
        avg = np.mean(emb_list, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        result[bid].append((avg.astype(np.float32), conf))
    return dict(result)


# ------------------------------------------------------------------
# Section filtering (same as infer_phase1)
# ------------------------------------------------------------------

def build_desc_to_sections(
    train_df: pd.DataFrame,
    products_df: pd.DataFrame,
    bundles_df: pd.DataFrame,
) -> Dict[str, Set[int]]:
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
    product_sections: Dict[str, Set[int]] = {}
    all_sections = {1, 2, 3}
    for _, row in products_df.iterrows():
        pid = str(row["product_asset_id"])
        desc = str(row["product_description"]).strip().upper()
        product_sections[pid] = desc_to_sections.get(desc, all_sections)
    return product_sections


def build_section_indices(
    encoded_pids: List[str],
    product_embeddings: np.ndarray,
    product_sections: Dict[str, Set[int]],
) -> Dict[int, Tuple[ProductIndex, List[str]]]:
    section_indices: Dict[int, Tuple[ProductIndex, List[str]]] = {}
    for section in [1, 2, 3]:
        mask = [i for i, pid in enumerate(encoded_pids) if section in product_sections.get(pid, {1, 2, 3})]
        sec_pids = [encoded_pids[i] for i in mask]
        sec_embs = product_embeddings[mask]
        sec_index = ProductIndex(sec_pids, sec_embs)
        section_indices[section] = (sec_index, sec_pids)
        print(f"    Section {section}: {len(sec_pids)} products")
    return section_indices


# ------------------------------------------------------------------
# Crop items
# ------------------------------------------------------------------

def build_crop_items(
    bundle_ids: List[str],
    bundle_image_map: Dict[str, Path],
    bundle_to_boxes: Dict[str, list],
    box_pad: float = 0.15,
) -> List[Tuple[str, Path, Any, float]]:
    items: List[Tuple[str, Path, Any, float]] = []
    for bid in bundle_ids:
        path = bundle_image_map.get(bid)
        if path is None or not path.exists():
            continue
        boxes = bundle_to_boxes.get(bid, [])
        if boxes:
            for x1, y1, x2, y2, conf in boxes:
                w, h = x2 - x1, y2 - y1
                px, py = int(w * box_pad), int(h * box_pad)
                items.append((bid, path, (x1 - px, y1 - py, x2 + px, y2 + py), conf))
        else:
            items.append((bid, path, None, 1.0))
    return items


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------

def retrieve(
    bundle_crops: Dict[str, List[Tuple[np.ndarray, float]]],
    section_indices: Dict[int, Tuple[ProductIndex, List[str]]],
    bundle_to_section: Dict[str, int],
    product_to_desc: Dict[str, str],
    top_k_per_crop: int = 5,
    max_products: int = 15,
    max_per_category: int = 2,
    bundle_to_gender: Optional[Dict[str, int]] = None,
    product_to_gender: Optional[Dict[str, int]] = None,
    gender_filter: bool = True,
) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    fallback_index = list(section_indices.values())[0]

    for bundle_id, crops in tqdm(bundle_crops.items(), desc="Retrieval", leave=False):
        section = bundle_to_section.get(bundle_id, 0)
        product_index, _ = section_indices.get(section, fallback_index)

        candidates: Dict[str, float] = {}
        for emb, conf in crops:
            indices, scores = product_index.search(emb, top_k_per_crop)
            for idx, sim in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                pid = product_index.product_ids[int(idx)]
                combined = float(sim) * max(float(conf), 0.1)
                candidates[pid] = max(candidates.get(pid, 0.0), combined)

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        # Gender filter
        if gender_filter and bundle_to_gender and product_to_gender:
            bg = bundle_to_gender.get(bundle_id, _GENDER_UNKNOWN)
            ranked = filter_cross_gender(ranked, bg, product_to_gender)

        # Category dedup
        final: List[str] = []
        cat_counts: Dict[str, int] = defaultdict(int)
        for pid, _ in ranked:
            if len(final) >= max_products:
                break
            desc = product_to_desc.get(pid, "")
            if desc and cat_counts[desc] >= max_per_category:
                continue
            final.append(pid)
            if desc:
                cat_counts[desc] += 1

        # Fill remaining slots ignoring category limit
        if len(final) < max_products:
            for pid, _ in ranked:
                if len(final) >= max_products:
                    break
                if pid not in final:
                    final.append(pid)

        results[bundle_id] = final
    return results


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

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
    device = resolve_device(cfg.params.device)
    batch_size = cfg.params.batch_size
    num_workers = cfg.params.num_workers
    amp = cfg.params.amp
    val_ratio = cfg.infer.val_ratio
    eval_ks = parse_ks(cfg.infer.eval_ks)
    checkpoint_path = to_absolute_path(cfg.infer.checkpoint_path)
    gender_filter = bool(getattr(cfg.infer, "gender_filter", True))

    TOP_K_PER_CROP = 5
    MAX_PRODUCTS = 15
    MAX_PER_CATEGORY = 2

    set_seed(cfg.params.seed)

    # ---- Load model ----
    print("=" * 60)
    print("INFER PHASE 1 — GR-Lite")
    print("=" * 60)
    model, input_size = load_grlite_inference_model(checkpoint_path, device)
    preprocess = build_preprocess(input_size)
    print(f"Device: {device} | AMP: {amp} | input_size: {input_size}")

    # ---- Load data ----
    bundles_df = pd.read_csv(bundles_csv)
    products_df = pd.read_csv(products_csv)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    bundle_image_map = build_image_map(bundle_images_dir)
    product_image_map = build_image_map(product_images_dir)

    all_product_ids = products_df["product_asset_id"].astype(str).tolist()
    product_ids = [pid for pid in all_product_ids if pid in product_image_map]
    product_to_desc: Dict[str, str] = {
        str(row.product_asset_id): str(row.product_description).strip().upper()
        for row in products_df.itertuples(index=False)
        if pd.notna(row.product_description)
    }
    print(f"Products: {len(product_ids)} with images / {len(all_product_ids)} total")

    # ---- Section mapping ----
    print("\n[0/5] Building section mapping...")
    desc_to_sections = build_desc_to_sections(train_df, products_df, bundles_df)
    product_sections = assign_product_sections(products_df, desc_to_sections)

    bundle_to_section: Dict[str, int] = {}
    for row in bundles_df.itertuples(index=False):
        bid = str(row.bundle_asset_id)
        sec = row.bundle_id_section
        if not pd.isna(sec):
            bundle_to_section[bid] = int(sec)

    # ---- Gender maps ----
    bundle_to_gender = load_bundle_genders(bundles_df)
    products_gender_csv = data_dir / "product_dataset_with_gender.csv"
    product_to_gender = load_product_genders(products_gender_csv)

    # ---- Encode products ----
    print("\n[1/5] Encoding products...")
    encoded_pids, product_embeddings = encode_products(
        product_ids, product_image_map, model, preprocess,
        device, batch_size, num_workers, amp,
    )
    print(f"  Product embeddings: {product_embeddings.shape}")

    # ---- Per-section indices ----
    print("\n[2/5] Building per-section product indices...")
    section_indices = build_section_indices(encoded_pids, product_embeddings, product_sections)

    # ---- Validation ----
    if val_ratio > 0:
        print(f"\n[3/5] Validation (val_ratio={val_ratio})...")
        train_bundle_ids = train_df["bundle_asset_id"].astype(str).tolist()
        unique_train_bundles = list(dict.fromkeys(train_bundle_ids))
        random.Random(cfg.params.seed).shuffle(unique_train_bundles)
        val_size = max(1, int(round(len(unique_train_bundles) * val_ratio)))
        val_bundle_ids = sorted(unique_train_bundles[:val_size])
        print(f"  Val bundles: {len(val_bundle_ids)}")

        gt_map: Dict[str, Set[str]] = defaultdict(set)
        for row in train_df.itertuples(index=False):
            gt_map[str(row.bundle_asset_id)].add(str(row.product_asset_id))

        val_boxes = detect_bundles(val_bundle_ids, bundle_image_map, cfg.params)
        val_crop_items = build_crop_items(val_bundle_ids, bundle_image_map, val_boxes)
        val_crop_embs = encode_bundle_crops(
            val_crop_items, model, preprocess, device, batch_size, num_workers, amp,
        )
        val_predictions = retrieve(
            val_crop_embs, section_indices, bundle_to_section,
            product_to_desc,
            top_k_per_crop=TOP_K_PER_CROP,
            max_products=MAX_PRODUCTS,
            max_per_category=MAX_PER_CATEGORY,
            bundle_to_gender=bundle_to_gender,
            product_to_gender=product_to_gender,
            gender_filter=gender_filter,
        )

        num_preds = [len(v) for v in val_predictions.values()]
        print(f"  Avg products/bundle: {np.mean(num_preds):.1f} "
              f"(min={min(num_preds)}, max={max(num_preds)})")

        val_metrics = validate(val_bundle_ids, val_predictions, gt_map, eval_ks)
        print("  Validation metrics:")
        for key, value in val_metrics.items():
            print(f"    {key}: {value:.6f}")

        metrics_out = output_dir / "val_metrics_grlite_phase1.json"
        metrics_out.write_text(json.dumps(val_metrics, indent=2))
    else:
        print("\n[3/5] Validation skipped")

    # ---- Test submission ----
    print(f"\n[4/5] Generating test submission...")
    test_bundle_ids = test_df["bundle_asset_id"].astype(str).drop_duplicates().tolist()
    print(f"  Test bundles: {len(test_bundle_ids)}")

    test_boxes = detect_bundles(test_bundle_ids, bundle_image_map, cfg.params)
    test_crop_items = build_crop_items(test_bundle_ids, bundle_image_map, test_boxes)
    test_crop_embs = encode_bundle_crops(
        test_crop_items, model, preprocess, device, batch_size, num_workers, amp,
    )
    test_predictions = retrieve(
        test_crop_embs, section_indices, bundle_to_section,
        product_to_desc,
        top_k_per_crop=TOP_K_PER_CROP,
        max_products=MAX_PRODUCTS,
        max_per_category=MAX_PER_CATEGORY,
        bundle_to_gender=bundle_to_gender,
        product_to_gender=product_to_gender,
        gender_filter=gender_filter,
    )

    submission_rows: List[Dict[str, str]] = []
    for bid in test_bundle_ids:
        for pid in test_predictions.get(bid, []):
            submission_rows.append({"bundle_asset_id": bid, "product_asset_id": pid})

    submission_df = pd.DataFrame(submission_rows, columns=["bundle_asset_id", "product_asset_id"])
    submission_out = output_dir / "test_submission_grlite_phase1.csv"
    submission_df.to_csv(submission_out, index=False)

    preds_per = [len(test_predictions.get(bid, [])) for bid in test_bundle_ids]
    print(f"\n{'=' * 60}")
    print(f"Submission saved: {submission_out} ({len(submission_df)} rows)")
    print(f"Bundles: {len(test_bundle_ids)} | Avg products/bundle: {np.mean(preds_per):.1f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
