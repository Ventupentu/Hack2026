"""Improved multi-object inference pipeline for bundle -> products retrieval."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import hydra
import hydra
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import InditexConfig
from src.detection import ClothingYOLODetector
from src.utils.metrics import evaluate_bundle_retrieval

cs = ConfigStore.instance()
cs.store(name="inditex_config", node=InditexConfig)

BoxXYXY = Tuple[int, int, int, int]
ScoredBox = Tuple[int, int, int, int, float]


class ProductDataset(Dataset):
    """Product dataset with image + optional text."""

    def __init__(
        self,
        product_ids: Sequence[str],
        image_map: Dict[str, Path],
        text_map: Dict[str, str],
        transform,
    ) -> None:
        self.product_ids = list(product_ids)
        self.image_map = image_map
        self.text_map = text_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.product_ids)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        product_id = self.product_ids[idx]
        img = open_image_safe(self.image_map[product_id])
        if img is None:
            return None
        return {
            "id": product_id,
            "img": self.transform(img),
            "text": self.text_map.get(product_id, ""),
        }


def collate_skip_none(batch: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Drop unreadable samples and collate tensors/lists."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    return {
        "id": [item["id"] for item in batch],
        "img": torch.stack([item["img"] for item in batch], dim=0),
        "text": [item["text"] for item in batch],
    }


def parse_ks(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("infer.eval_ks must contain at least one positive integer.")
        raise ValueError("infer.eval_ks must contain at least one positive integer.")
    return sorted(set(values))


def build_image_map(image_dir: Path) -> Dict[str, Path]:
    """Map asset_id (filename stem) to local image path."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    image_map: Dict[str, Path] = {}
    for path in image_dir.iterdir():
        if path.is_file():
            image_map[path.stem] = path
    return image_map


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def split_val_bundles(bundle_ids: Sequence[str], val_ratio: float, seed: int) -> List[str]:
    if val_ratio <= 0:
        return []
    ids = list(dict.fromkeys(bundle_ids))
    random.Random(seed).shuffle(ids)
    val_size = max(1, int(round(len(ids) * val_ratio)))
    return sorted(ids[:val_size])


def build_gt_map(train_df: pd.DataFrame) -> Dict[str, Set[str]]:
    gt_map: Dict[str, Set[str]] = defaultdict(set)
    for row in train_df.itertuples(index=False):
        gt_map[str(row.bundle_asset_id)].add(str(row.product_asset_id))
    return gt_map


def open_image_safe(path: Path, retries: int = 1) -> Optional[Image.Image]:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
            last_err = exc
    print(f"Warning: failed to read image {path} ({last_err})")
    return None


def _normalize_scored_box(value: Any) -> Optional[ScoredBox]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
        score = float(value[4]) if len(value) >= 5 else 1.0
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2, score)


def load_boxes_cache(path: Path) -> Dict[str, List[ScoredBox]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, List[ScoredBox]] = {}
    for bundle_id, boxes in payload.items():
        if not isinstance(bundle_id, str) or not isinstance(boxes, list):
            continue
        clean: List[ScoredBox] = []
        for box in boxes:
            parsed = _normalize_scored_box(box)
            if parsed is not None:
                clean.append(parsed)
        out[bundle_id] = clean
    return out


def save_boxes_cache(path: Path, data: Dict[str, List[ScoredBox]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {bid: [list(box) for box in boxes] for bid, boxes in data.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def box_iou(a: BoxXYXY, b: BoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def nms_scored_boxes(
    boxes: Sequence[ScoredBox],
    iou_threshold: float,
    max_boxes: int,
    min_score: float,
) -> List[ScoredBox]:
    filtered = [box for box in boxes if box[4] >= min_score]
    filtered.sort(key=lambda x: x[4], reverse=True)
    keep: List[ScoredBox] = []
    for candidate in filtered:
        cand_xyxy = candidate[:4]
        if all(box_iou(cand_xyxy, kept[:4]) < iou_threshold for kept in keep):
            keep.append(candidate)
        if len(keep) >= max_boxes:
            break
    return keep


def expand_box(box: BoxXYXY, width: int, height: int, padding: float) -> BoxXYXY:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    pad_w = int(math.ceil(w * padding))
    pad_h = int(math.ceil(h * padding))
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(width, x2 + pad_w)
    ny2 = min(height, y2 + pad_h)
    if nx2 <= nx1 or ny2 <= ny1:
        return box
    return (nx1, ny1, nx2, ny2)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


@torch.inference_mode()
def encode_product_index(
    clip_model: torch.nn.Module,
    tokenizer,
    product_ids: Sequence[str],
    product_image_map: Dict[str, Path],
    product_text_map: Dict[str, str],
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> Tuple[List[str], torch.Tensor]:
    dataset = ProductDataset(
        product_ids=product_ids,
        image_map=product_image_map,
        text_map=product_text_map,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )

    encoded_ids: List[str] = []
    encoded_embs: List[torch.Tensor] = []
    amp_enabled = amp and device.type == "cuda"
    clip_model.eval()
    for batch in tqdm(loader, desc="Encoding products", leave=False):
        if batch is None:
            continue
        imgs = batch["img"].to(device, non_blocking=True)
        texts = batch["text"]
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            image_feats = clip_model.encode_image(imgs).float()
            text_rows = [i for i, txt in enumerate(texts) if isinstance(txt, str) and txt.strip()]
            if text_rows:
                tokens = tokenizer([texts[i] for i in text_rows]).to(device, non_blocking=True)
                text_feats = clip_model.encode_text(tokens).float()
                image_feats[text_rows] = image_feats[text_rows] + text_feats
            feats = F.normalize(image_feats, p=2, dim=1)

        encoded_ids.extend(batch["id"])
        encoded_embs.append(feats)

    if not encoded_embs:
        raise RuntimeError("No product embeddings could be encoded.")
    return encoded_ids, torch.cat(encoded_embs, dim=0)


def detect_boxes_for_bundle_ids(
    bundle_ids: Sequence[str],
    bundle_image_map: Dict[str, Path],
    detector: ClothingYOLODetector,
    cache_path: Path,
    nms_iou_threshold: float,
    max_boxes_per_image: int,
    min_box_score: float,
) -> Dict[str, List[ScoredBox]]:
    cache = load_boxes_cache(cache_path)
    missing_ids = [bid for bid in bundle_ids if bid not in cache]
    if missing_ids:
        for bundle_id in tqdm(missing_ids, desc="Detecting bundle boxes"):
            image_path = bundle_image_map.get(bundle_id)
            if image_path is None or not image_path.exists():
                cache[bundle_id] = []
                continue
            raw_boxes = detector.detect_boxes(image_path)
            cache[bundle_id] = nms_scored_boxes(
                boxes=raw_boxes,
                iou_threshold=nms_iou_threshold,
                max_boxes=max_boxes_per_image,
                min_score=min_box_score,
            )
        save_boxes_cache(cache_path, cache)
        print(f"Saved bundle bbox cache: {cache_path}")
    else:
        print(f"Loaded bundle bbox cache: {cache_path}")

    return {bundle_id: cache.get(bundle_id, []) for bundle_id in bundle_ids}


@torch.inference_mode()
def predict_bundle_topk(
    bundle_id: str,
    bundle_image_map: Dict[str, Path],
    bundle_boxes_map: Dict[str, List[ScoredBox]],
    clip_model: torch.nn.Module,
    preprocess_val,
    product_ids: Sequence[str],
    product_embeddings: torch.Tensor,
    device: torch.device,
    amp: bool,
    retrieval_topk: int,
    max_products_per_box: int,
    box_padding: float,
    final_k: int,
    fallback_products: Sequence[str],
) -> List[str]:
    image_path = bundle_image_map.get(bundle_id)
    if image_path is None:
        return list(fallback_products[:final_k])

    image = open_image_safe(image_path)
    if image is None:
        return list(fallback_products[:final_k])

    width, height = image.size
    scored_boxes = bundle_boxes_map.get(bundle_id, [])
    if not scored_boxes:
        scored_boxes = [(0, 0, width, height, 1.0)]

    crop_tensors: List[torch.Tensor] = []
    for x1, y1, x2, y2, _score in scored_boxes:
        ex1, ey1, ex2, ey2 = expand_box((x1, y1, x2, y2), width=width, height=height, padding=box_padding)
        crop = image.crop((ex1, ey1, ex2, ey2))
        crop_tensors.append(preprocess_val(crop))

    if not crop_tensors:
        return list(fallback_products[:final_k])

    crops = torch.stack(crop_tensors, dim=0).to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        crop_embs = clip_model.encode_image(crops).float()
        crop_embs = F.normalize(crop_embs, p=2, dim=1)

    k = min(retrieval_topk, product_embeddings.shape[0])
    if k <= 0:
        return list(fallback_products[:final_k])

    sims = crop_embs @ product_embeddings.T
    top_scores, top_indices = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)

    fused_scores: Dict[str, float] = {}
    per_box_cap = min(max_products_per_box, k)
    for row in range(top_indices.shape[0]):
        for col in range(per_box_cap):
            product_idx = int(top_indices[row, col].item())
            score = float(top_scores[row, col].item())
            product_id = product_ids[product_idx]
            if product_id not in fused_scores or score > fused_scores[product_id]:
                fused_scores[product_id] = score

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    preds = [pid for pid, _ in ranked[:final_k]]
    if len(preds) < final_k:
        seen = set(preds)
        for pid in fallback_products:
            if pid in seen:
                continue
            preds.append(pid)
            seen.add(pid)
            if len(preds) >= final_k:
                break
    return preds[:final_k]


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: InditexConfig) -> None:
    files = cfg.files
    params = cfg.params
    infer_cfg = cfg.infer

    data_dir = Path(to_absolute_path(files.data_dir))
    bundles_csv = data_dir / "bundles_dataset.csv"
    products_csv = data_dir / "product_dataset.csv"
    train_csv = data_dir / "bundles_product_match_train.csv"
    test_csv = data_dir / "bundles_product_match_test.csv"
    bundle_images_dir = Path(to_absolute_path(files.bundles_images))
    product_images_dir = Path(to_absolute_path(files.products_images))

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_out = output_dir / "test_submission.csv"
    metrics_out = output_dir / "val_metrics.json"

    val_ratio = float(infer_cfg.val_ratio)
    eval_ks = parse_ks(infer_cfg.eval_ks)
    top_n_submit = min(int(infer_cfg.top_n_submit), 15)
    if top_n_submit <= 0:
        raise ValueError("infer.top_n_submit must be > 0")

    retrieval_topk = int(getattr(infer_cfg, "retrieval_topk", 100))
    max_products_per_box = int(getattr(infer_cfg, "max_products_per_box", 5))
    max_boxes_per_image = int(getattr(infer_cfg, "max_boxes_per_image", 10))
    box_padding = float(getattr(infer_cfg, "box_padding", 0.15))
    nms_iou_threshold = float(getattr(infer_cfg, "nms_iou_threshold", 0.5))
    min_box_score = float(getattr(infer_cfg, "min_box_score", 0.15))
    checkpoint_path = str(getattr(infer_cfg, "checkpoint_path", "")).strip()
    boxes_cache_path = str(getattr(infer_cfg, "boxes_cache_path", "")).strip()
    detector_conf_threshold = float(
        getattr(infer_cfg, "detector_conf_threshold", params.bbox_conf_threshold)
    )
    detector_iou_threshold = float(
        getattr(infer_cfg, "detector_iou_threshold", params.bbox_iou_threshold)
    )

    if max_products_per_box <= 0:
        raise ValueError("infer.max_products_per_box must be > 0")
    if retrieval_topk <= 0:
        raise ValueError("infer.retrieval_topk must be > 0")
    if max_boxes_per_image <= 0:
        raise ValueError("infer.max_boxes_per_image must be > 0")

    seed = int(params.seed)
    set_seed(seed)
    device = resolve_device(params.device)
    amp_enabled = bool(params.amp and device.type == "cuda")

    bundles_df = pd.read_csv(bundles_csv)
    products_df = pd.read_csv(products_csv)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    bundle_image_map = build_image_map(bundle_images_dir)
    product_image_map = build_image_map(product_images_dir)
    bundle_image_map = build_image_map(bundle_images_dir)
    product_image_map = build_image_map(product_images_dir)

    product_ids_all = products_df["product_asset_id"].astype(str).tolist()
    product_ids = [pid for pid in product_ids_all if pid in product_image_map]
    if not product_ids:
        raise RuntimeError("No product images found for product ids.")

    product_text_map = {
        str(row.product_asset_id): str(row.product_description)
        if not pd.isna(row.product_description)
        else ""
        for row in products_df.itertuples(index=False)
    }

    clip_model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    clip_model = clip_model.to(device).eval()

    if checkpoint_path:
        ckpt = Path(to_absolute_path(checkpoint_path))
        if not ckpt.exists():
            raise FileNotFoundError(f"infer.checkpoint_path does not exist: {ckpt}")
        payload = torch.load(ckpt, map_location=device)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Invalid checkpoint format at {ckpt}")
        state_dict = strip_module_prefix(state_dict)
        missing, unexpected = clip_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {ckpt}")
        print(f"Checkpoint compatibility | missing={len(missing)} unexpected={len(unexpected)}")

    print(f"Encoding {len(product_ids)} products on {device}...")
    encoded_product_ids, product_embeddings = encode_product_index(
        clip_model=clip_model,
        tokenizer=tokenizer,
        product_ids=product_ids,
        product_image_map=product_image_map,
        product_text_map=product_text_map,
        transform=preprocess_val,
        device=device,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        amp=amp_enabled,
    )
    product_ids = encoded_product_ids
    print(f"Product index shape: {tuple(product_embeddings.shape)}")

    train_bundle_ids = train_df["bundle_asset_id"].astype(str).tolist()
    val_bundle_ids = split_val_bundles(train_bundle_ids, val_ratio=val_ratio, seed=seed)
    val_bundle_ids = split_val_bundles(train_bundle_ids, val_ratio=val_ratio, seed=seed)
    gt_map = build_gt_map(train_df)

    popular_products = train_df["product_asset_id"].astype(str).tolist()
    fallback_products = [pid for pid, _ in Counter(popular_products).most_common()]
    if not fallback_products:
        fallback_products = list(product_ids)
    seen_fallback = set(fallback_products)
    for pid in product_ids:
        if pid not in seen_fallback:
            fallback_products.append(pid)
            seen_fallback.add(pid)

    query_bundle_ids = sorted(
        set(val_bundle_ids) | set(test_df["bundle_asset_id"].astype(str).drop_duplicates().tolist())
    )
    bundle_boxes_map: Dict[str, List[ScoredBox]] = {}
    if params.use_bundle_boxes:
        cache_path = (
            Path(to_absolute_path(boxes_cache_path))
            if boxes_cache_path
            else Path(to_absolute_path(files.yolo_detections_dir)) / "infer_bundle_boxes_cache.json"
        )
        detector = ClothingYOLODetector(
            model_id=params.bbox_model_id,
            conf_threshold=detector_conf_threshold,
            iou_threshold=detector_iou_threshold,
            max_boxes_per_image=max_boxes_per_image,
            min_area_ratio=params.bbox_min_area_ratio,
        )
        bundle_boxes_map = detect_boxes_for_bundle_ids(
            bundle_ids=query_bundle_ids,
            bundle_image_map=bundle_image_map,
            detector=detector,
            cache_path=cache_path,
            nms_iou_threshold=nms_iou_threshold,
            max_boxes_per_image=max_boxes_per_image,
            min_box_score=min_box_score,
        )

    val_metrics: Dict[str, float] = {"num_bundles_evaluated": 0.0}
    if val_bundle_ids:
        max_eval_k = max(eval_ks)
        val_predictions: List[List[str]] = []
        for bundle_id in tqdm(val_bundle_ids, desc="Val inference"):
            preds = predict_bundle_topk(
                bundle_id=bundle_id,
                bundle_image_map=bundle_image_map,
                bundle_boxes_map=bundle_boxes_map,
                clip_model=clip_model,
                preprocess_val=preprocess_val,
                product_ids=product_ids,
                product_embeddings=product_embeddings,
                device=device,
                amp=amp_enabled,
                retrieval_topk=retrieval_topk,
                max_products_per_box=max_products_per_box,
                box_padding=box_padding,
                final_k=max_eval_k,
                fallback_products=fallback_products,
            )
            val_predictions.append(preds)

        val_metrics = evaluate_bundle_retrieval(
            bundle_ids=val_bundle_ids,
            predictions=val_predictions,
            ground_truth=gt_map,
            ks=eval_ks,
        )
        print("Validation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")

    test_bundle_ids = test_df["bundle_asset_id"].astype(str).drop_duplicates().tolist()
    known_bundle_ids = set(bundles_df["bundle_asset_id"].astype(str).tolist())
    unknown_test_bundles = sum(1 for bid in test_bundle_ids if bid not in known_bundle_ids)
    if unknown_test_bundles:
        print(f"Warning: {unknown_test_bundles} test bundle ids not found in bundles CSV.")

    test_predictions: Dict[str, List[str]] = {}
    for bundle_id in tqdm(test_bundle_ids, desc="Test inference"):
        preds = predict_bundle_topk(
            bundle_id=bundle_id,
            bundle_image_map=bundle_image_map,
            bundle_boxes_map=bundle_boxes_map,
            clip_model=clip_model,
            preprocess_val=preprocess_val,
            product_ids=product_ids,
            product_embeddings=product_embeddings,
            device=device,
            amp=amp_enabled,
            retrieval_topk=retrieval_topk,
            max_products_per_box=max_products_per_box,
            box_padding=box_padding,
            final_k=top_n_submit,
            fallback_products=fallback_products,
        )
        test_predictions[bundle_id] = preds

    submission_rows: List[Dict[str, str]] = []
    for bundle_id in test_bundle_ids:
        for product_id in test_predictions[bundle_id][:top_n_submit]:
            submission_rows.append(
                {"bundle_asset_id": bundle_id, "product_asset_id": product_id}
            )

    submission_df = pd.DataFrame(submission_rows, columns=["bundle_asset_id", "product_asset_id"])
    submission_df.to_csv(submission_out, index=False)
    submission_df.to_csv(submission_out, index=False)

    summary = {
        "model": "hf-hub:Marqo/marqo-fashionSigLIP",
        "device": str(device),
        "amp": bool(amp_enabled),
        "checkpoint_path": checkpoint_path,
        "num_products_indexed": int(len(product_ids)),
        "num_test_bundles": int(len(test_bundle_ids)),
        "rows_written_submission": int(len(submission_df)),
        "use_bundle_boxes": bool(params.use_bundle_boxes),
        "retrieval_topk": int(retrieval_topk),
        "max_products_per_box": int(max_products_per_box),
        "max_boxes_per_image": int(max_boxes_per_image),
        "box_padding": float(box_padding),
        "nms_iou_threshold": float(nms_iou_threshold),
        "min_box_score": float(min_box_score),
        "val_metrics": val_metrics,
    }
    metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved submission: {submission_out} ({len(submission_df)} rows)")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()