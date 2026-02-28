"""OpenCLIP retrieval training implementation (bundle -> products)."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import InditexConfig
from src.detection import BoxXYXY, ClothingYOLODetector, detect_boxes_for_assets


class OpenCLIPMultimodalEncoder(nn.Module):
    """Wrapper exposing multimodal forward for DataParallel.

    Includes:
      - Learnable fusion gate (alpha) for weighted image+text combination
      - Learnable log-temperature for contrastive loss
    """

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.clip_model = clip_model
        # Learnable fusion gate: alpha * image + (1-alpha) * text
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        # Learnable temperature for contrastive loss (ln(1/0.07) ≈ 2.659)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    @property
    def temperature(self) -> torch.Tensor:
        """Clamped temperature value."""
        return (1.0 / self.log_temperature.exp()).clamp(min=0.01, max=0.5)

    def forward(self, images: torch.Tensor, text: Optional[torch.Tensor] = None) -> torch.Tensor:
        if text is not None:
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(text)
            alpha = torch.sigmoid(self.fusion_alpha)
            return alpha * image_features + (1 - alpha) * text_features
        return self.clip_model.encode_image(images)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device."""
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def ensure_dir(path: Path) -> None:
    """Create folder if needed."""
    path.mkdir(parents=True, exist_ok=True)


def _normalize_box(value: Any) -> Optional[BoxXYXY]:
    """Parse and validate one XYXY box from cache/json."""
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def load_boxes_cache(path: Path) -> Dict[str, List[BoxXYXY]]:
    """Load cached bundle boxes from json."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, List[BoxXYXY]] = {}
    for bundle_id, boxes in payload.items():
        if not isinstance(bundle_id, str) or not isinstance(boxes, list):
            continue
        clean_boxes: List[BoxXYXY] = []
        for box in boxes:
            norm = _normalize_box(box)
            if norm is not None:
                clean_boxes.append(norm)
        out[bundle_id] = clean_boxes
    return out


def save_boxes_cache(path: Path, bundle_to_boxes: Dict[str, List[BoxXYXY]]) -> None:
    """Write bundle boxes cache to json."""
    ensure_dir(path.parent)
    payload = {
        bid: [list(box) for box in boxes]
        for bid, boxes in bundle_to_boxes.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    """Read jsonl or csv manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest extension: {path.suffix}. Use .jsonl or .csv")


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _parse_id_list(value: Any) -> List[str]:
    """Parse list-like product ids from str/list/json."""
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_str(v) for v in value if _as_str(v)]
    text = _as_str(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [_as_str(v) for v in parsed if _as_str(v)]
        except json.JSONDecodeError:
            pass
    for sep in ("|", ",", ";", " "):
        if sep in text:
            parts = [_as_str(x) for x in text.split(sep)]
            cleaned = [x for x in parts if x]
            if cleaned:
                return cleaned
    return [text]


def _first_existing_key(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row and _as_str(row.get(key)):
            return _as_str(row.get(key))
    return ""


def _default_bundle_path(bundle_id: str, bundles_images_dir: Path) -> Path:
    return (bundles_images_dir / f"{bundle_id}.jpg").resolve()


def _default_product_path(product_id: str, products_images_dir: Path) -> Path:
    return (products_images_dir / f"{product_id}.jpg").resolve()


def parse_products_manifest(path: Path, products_images_dir: Path) -> Tuple[Dict[str, Path], Dict[str, str]]:
    """Return product_id -> image_path map and product_id -> text map."""
    rows = read_manifest_rows(path)
    product_to_image: Dict[str, Path] = {}
    product_to_text: Dict[str, str] = {}
    for row in rows:
        pid = _first_existing_key(row, ("product_asset_id", "product_id", "asset_id", "id"))
        if not pid:
            continue
        image_path = _first_existing_key(
            row,
            ("image_path", "product_image_path", "path", "local_image_path"),
        )
        if not image_path:
            image_path = str(_default_product_path(pid, products_images_dir))
        product_to_image[pid] = Path(image_path).expanduser().resolve()
        product_to_text[pid] = _first_existing_key(row, ("product_description", "description", "text"))
    if not product_to_image:
        raise RuntimeError("No product entries found in products_manifest.")
    return product_to_image, product_to_text


def parse_bundle_manifest(
    path: Path,
    product_to_image: Dict[str, Path],
    bundles_images_dir: Path,
) -> Tuple[Dict[str, Path], Dict[str, Set[str]]]:
    """Parse train/val manifest into bundle_image and positives mapping."""
    rows = read_manifest_rows(path)
    bundle_to_image: Dict[str, Path] = {}
    bundle_to_products: Dict[str, Set[str]] = defaultdict(set)

    for row in rows:
        bundle_id = _first_existing_key(row, ("bundle_asset_id", "bundle_id", "query_id", "id"))
        if not bundle_id:
            continue

        bundle_img = _first_existing_key(
            row,
            ("bundle_image_path", "image_path", "query_image_path", "path"),
        )
        bundle_to_image[bundle_id] = (
            Path(bundle_img).expanduser().resolve()
            if bundle_img
            else _default_bundle_path(bundle_id, bundles_images_dir)
        )

        direct_pid = _first_existing_key(row, ("product_asset_id", "product_id", "candidate_id"))
        if direct_pid:
            bundle_to_products[bundle_id].add(direct_pid)

        for key in ("product_asset_ids", "product_ids", "positives", "positive_product_ids"):
            if key in row:
                for pid in _parse_id_list(row.get(key)):
                    bundle_to_products[bundle_id].add(pid)

    filtered: Dict[str, Set[str]] = {}
    dropped = 0
    for bid, pids in bundle_to_products.items():
        keep = {pid for pid in pids if pid in product_to_image}
        dropped += len(pids) - len(keep)
        if keep:
            filtered[bid] = keep
    if dropped > 0:
        print(f"Warning: dropped {dropped} bundle-product links missing in products_manifest.")

    bundle_to_image = {bid: bundle_to_image[bid] for bid in filtered.keys() if bid in bundle_to_image}
    if not filtered:
        raise RuntimeError(f"No valid bundle->product links found in manifest: {path}")
    return bundle_to_image, filtered


def detect_bundle_boxes_with_cache(
    bundle_to_image: Dict[str, Path],
    output_dir: Path,
    model_id: str,
    conf_threshold: float,
    iou_threshold: float,
    max_boxes_per_image: int,
    min_area_ratio: float,
    cache_path: str,
) -> Dict[str, List[BoxXYXY]]:
    """Detect and cache XYXY boxes for each bundle image."""
    resolved_cache = (
        Path(cache_path).expanduser().resolve()
        if cache_path
        else (output_dir / "bundle_boxes_cache.json").resolve()
    )

    bundle_to_boxes = load_boxes_cache(resolved_cache)
    missing = {
        bundle_id: image_path
        for bundle_id, image_path in bundle_to_image.items()
        if bundle_id not in bundle_to_boxes
    }

    if missing:
        print(f"Detecting boxes for {len(missing)} bundles...")
        try:
            detector = ClothingYOLODetector(
                model_id=model_id,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_boxes_per_image=max_boxes_per_image,
                min_area_ratio=min_area_ratio,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "params.use_bundle_boxes=true requiere ultralyticsplus. "
                "Instala con: pip install ultralyticsplus"
            ) from exc

        detected = detect_boxes_for_assets(detector, missing, show_progress=True)
        bundle_to_boxes.update(detected)
        save_boxes_cache(resolved_cache, bundle_to_boxes)
        print(f"Saved bbox cache: {resolved_cache}")
    else:
        print(f"Loaded bbox cache: {resolved_cache}")

    return {bundle_id: bundle_to_boxes.get(bundle_id, []) for bundle_id in bundle_to_image}


def open_image_safe(path: Path, retries: int = 1) -> Optional[Image.Image]:
    """Safely open an image with small retry count."""
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
            last_err = exc
    print(f"Warning: failed to read image {path} ({last_err})")
    return None


def crop_with_box(image: Image.Image, box: BoxXYXY) -> Image.Image:
    """Crop image with bounds-safe XYXY coordinates."""
    x1, y1, x2, y2 = box
    width, height = image.size
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


class BundleMultiPositiveDataset(Dataset):
    """One sample per bundle, returning ALL positive products for multi-positive loss.

    Also supports hard negatives: if hard_negatives dict is provided,
    includes the pre-mined hard negative product images in the batch.
    """

    def __init__(
        self,
        bundle_to_image: Dict[str, Path],
        bundle_to_products: Dict[str, Set[str]],
        product_to_image: Dict[str, Path],
        product_to_text: Dict[str, str],
        bundle_transform: Callable[[Image.Image], torch.Tensor],
        product_transform: Callable[[Image.Image], torch.Tensor],
        tokenizer: Any,
        max_positives: int = 8,
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        max_hard_negatives: int = 4,
    ) -> None:
        self.bundle_ids = sorted(bundle_to_products.keys())
        self.bundle_to_image = bundle_to_image
        self.bundle_to_products = {k: sorted(v) for k, v in bundle_to_products.items()}
        self.product_to_image = product_to_image
        self.product_to_text = product_to_text
        self.bundle_transform = bundle_transform
        self.product_transform = product_transform
        self.tokenizer = tokenizer
        self.max_positives = max_positives
        self.hard_negatives = hard_negatives or {}
        self.max_hard_negatives = max_hard_negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        bundle_id = self.bundle_ids[idx]
        all_product_ids = self.bundle_to_products[bundle_id]

        # Sample up to max_positives from the positive set
        if len(all_product_ids) > self.max_positives:
            pos_ids = random.sample(all_product_ids, self.max_positives)
        else:
            pos_ids = list(all_product_ids)

        bundle_img = open_image_safe(self.bundle_to_image[bundle_id])
        if bundle_img is None:
            return None

        pos_imgs: List[torch.Tensor] = []
        pos_texts: List[torch.Tensor] = []
        valid_pos_ids: List[str] = []
        for pid in pos_ids:
            img = open_image_safe(self.product_to_image[pid])
            if img is None:
                continue
            pos_imgs.append(self.product_transform(img))
            valid_pos_ids.append(pid)
            text = self.product_to_text.get(pid, "")
            if text:
                pos_texts.append(self.tokenizer(text).squeeze(0))
            else:
                pos_texts.append(None)

        if not pos_imgs:
            return None

        # Hard negatives
        neg_imgs: List[torch.Tensor] = []
        neg_texts: List[torch.Tensor] = []
        neg_ids: List[str] = []
        hn_list = self.hard_negatives.get(bundle_id, [])
        hn_sample = hn_list[:self.max_hard_negatives]
        for nid in hn_sample:
            if nid in self.product_to_image:
                img = open_image_safe(self.product_to_image[nid])
                if img is None:
                    continue
                neg_imgs.append(self.product_transform(img))
                neg_ids.append(nid)
                text = self.product_to_text.get(nid, "")
                if text:
                    neg_texts.append(self.tokenizer(text).squeeze(0))
                else:
                    neg_texts.append(None)

        return {
            "bundle_id": bundle_id,
            "bundle_img": self.bundle_transform(bundle_img),
            "pos_imgs": pos_imgs,         # List[Tensor]
            "pos_texts": pos_texts,       # List[Optional[Tensor]]
            "pos_ids": valid_pos_ids,     # List[str]
            "neg_imgs": neg_imgs,         # List[Tensor]
            "neg_texts": neg_texts,       # List[Optional[Tensor]]
            "neg_ids": neg_ids,           # List[str]
            "num_pos": len(pos_imgs),
            "num_neg": len(neg_imgs),
        }


def collate_multi_positive(batch: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Collate for multi-positive batches with variable numbers of positives/negatives.

    Returns:
        bundle_imgs: [B, C, H, W]
        product_imgs: [N_total, C, H, W]  (all positives + hard negatives concatenated)
        product_texts: [N_total, context_len] or None
        pos_mask: [B, N_total] boolean — True where product is a positive for that bundle
        bundle_ids, product_ids: lists for debugging
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    bundle_imgs = torch.stack([item["bundle_img"] for item in batch], dim=0)

    # Flatten all product images (positives + hard negatives) into one big tensor
    all_product_imgs: List[torch.Tensor] = []
    all_product_texts: List[Optional[torch.Tensor]] = []
    all_product_ids: List[str] = []
    # pos_mask[i, j] = True if product j is a positive for bundle i
    pos_ranges: List[Tuple[int, int]] = []  # (start, end) of positives for each bundle

    offset = 0
    for item in batch:
        n_pos = item["num_pos"]
        n_neg = item["num_neg"]
        all_product_imgs.extend(item["pos_imgs"])
        all_product_texts.extend(item["pos_texts"])
        all_product_ids.extend(item["pos_ids"])
        pos_ranges.append((offset, offset + n_pos))
        offset += n_pos
        all_product_imgs.extend(item["neg_imgs"])
        all_product_texts.extend(item["neg_texts"])
        all_product_ids.extend(item["neg_ids"])
        offset += n_neg

    product_imgs = torch.stack(all_product_imgs, dim=0) if all_product_imgs else None

    # Build text tensor if any texts exist
    has_any_text = any(t is not None for t in all_product_texts)
    product_texts = None
    if has_any_text and all_product_texts:
        # Find a valid text tensor to get the shape
        text_shape = None
        for t in all_product_texts:
            if t is not None:
                text_shape = t.shape
                break
        if text_shape is not None:
            text_list = []
            for t in all_product_texts:
                if t is not None:
                    text_list.append(t)
                else:
                    text_list.append(torch.zeros(text_shape, dtype=torch.long))
            product_texts = torch.stack(text_list, dim=0)

    # Build pos_mask
    B = len(batch)
    N = len(all_product_imgs)
    pos_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, (start, end) in enumerate(pos_ranges):
        pos_mask[i, start:end] = True

    return {
        "bundle_imgs": bundle_imgs,
        "product_imgs": product_imgs,
        "product_texts": product_texts,
        "pos_mask": pos_mask,
        "bundle_ids": [item["bundle_id"] for item in batch],
        "product_ids": all_product_ids,
    }


class AssetImageDataset(Dataset):
    """Simple asset image dataset for encoding."""

    def __init__(self, ids: Sequence[str], id_to_path: Dict[str, Path], transform, id_to_text: Optional[Dict[str, str]] = None, tokenizer: Any = None) -> None:
        self.ids = list(ids)
        self.id_to_path = id_to_path
        self.transform = transform
        self.id_to_text = id_to_text
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        asset_id = self.ids[idx]
        img = open_image_safe(self.id_to_path[asset_id])
        if img is None:
            return None
        out = {"id": asset_id, "img": self.transform(img)}
        if self.id_to_text is not None and self.tokenizer is not None:
            text = self.id_to_text.get(asset_id, "")
            if text:
                out["text"] = self.tokenizer(text).squeeze(0)
        return out


def collate_skip_none(batch: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Collate function that skips unreadable samples."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def compute_hard_negatives(
    model: nn.Module,
    preprocess_val: Any,
    device: torch.device,
    amp: bool,
    bundle_to_image: Dict[str, Path],
    bundle_to_products: Dict[str, Set[str]],
    product_to_image: Dict[str, Path],
    product_to_text: Dict[str, str],
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    top_k: int = 16,
) -> Dict[str, List[str]]:
    """Pre-compute gallery embeddings and mine top-K hard negatives per bundle.

    Hard negatives are the closest products in the embedding space that are NOT
    in the positive set for each bundle.
    """
    print("Mining hard negatives from full gallery...")
    t0 = time.time()

    product_ids_list = sorted(product_to_image.keys())
    product_loader = DataLoader(
        AssetImageDataset(product_ids_list, product_to_image, preprocess_val,
                          id_to_text=product_to_text, tokenizer=tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    encoded_product_ids, product_embs = encode_images(model, product_loader, device, amp)

    bundle_ids_list = sorted(bundle_to_products.keys())
    bundle_loader = DataLoader(
        AssetImageDataset(bundle_ids_list, bundle_to_image, preprocess_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    encoded_bundle_ids, bundle_embs = encode_images(model, bundle_loader, device, amp)

    pid_to_idx = {pid: i for i, pid in enumerate(encoded_product_ids)}

    hard_neg_dict: Dict[str, List[str]] = {}
    # Process in batches to avoid OOM on large galleries
    for start in range(0, len(encoded_bundle_ids), batch_size):
        end = min(start + batch_size, len(encoded_bundle_ids))
        batch_ids = encoded_bundle_ids[start:end]
        batch_emb = bundle_embs[start:end]
        sims = batch_emb @ product_embs.T  # [B, N_products]

        # Get top-(top_k + max_positives) to have room after filtering positives
        topk_val = min(top_k + 20, sims.shape[1])
        _, topk_idx = torch.topk(sims, k=topk_val, dim=1, largest=True, sorted=True)

        for row, bid in enumerate(batch_ids):
            pos_pids = bundle_to_products.get(bid, set())
            pos_indices = {pid_to_idx[pid] for pid in pos_pids if pid in pid_to_idx}
            hard_negs: List[str] = []
            for col_idx in topk_idx[row].tolist():
                if col_idx not in pos_indices:
                    hard_negs.append(encoded_product_ids[col_idx])
                    if len(hard_negs) >= top_k:
                        break
            hard_neg_dict[bid] = hard_negs

    elapsed = time.time() - t0
    print(f"Hard negative mining done in {elapsed:.1f}s — mined for {len(hard_neg_dict)} bundles")
    return hard_neg_dict


def multi_positive_nce_loss(
    bundle_embs: torch.Tensor,
    product_embs: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Multi-positive NCE loss.

    L = -log( sum(exp(s_pos/τ)) / sum(exp(s_all/τ)) )  per bundle, averaged.

    Args:
        bundle_embs: [B, D] normalized bundle embeddings
        product_embs: [N, D] normalized product embeddings (positives + negatives)
        pos_mask: [B, N] boolean mask, True for positives
        temperature: scalar or Tensor
    """
    # Similarity matrix [B, N]
    logits = (bundle_embs @ product_embs.T) / temperature

    # For numerical stability
    logits_max = logits.max(dim=1, keepdim=True).values
    logits = logits - logits_max

    exp_logits = torch.exp(logits)

    # Sum of exp over all products
    log_denom = torch.log(exp_logits.sum(dim=1) + 1e-8)  # [B]

    # Sum of exp over positive products only
    pos_exp = (exp_logits * pos_mask.float()).sum(dim=1)
    log_pos_sum = torch.log(pos_exp + 1e-8)  # [B]

    # Loss per bundle: -log(sum_pos / sum_all)
    loss = -(log_pos_sum - log_denom)

    return loss.mean()


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
    """Cosine scheduler with 10% warmup."""
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def encode_images(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[List[str], torch.Tensor]:
    """Encode dataset images into normalized embeddings."""
    all_ids: List[str] = []
    all_embs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            if batch is None:
                continue
            imgs = batch["img"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                if "text" in batch and batch["text"] is not None:
                    # Provide texts if available for computing text+image features
                    texts = batch["text"].to(device, non_blocking=True)
                    feats = model(imgs, text=texts)
                else:
                    feats = model(imgs)
            feats = F.normalize(feats.float(), p=2, dim=1)
            all_ids.extend(batch["id"])
            all_embs.append(feats)

    if not all_embs:
        return [], torch.empty((0, 0), dtype=torch.float32, device=device)
    return all_ids, torch.cat(all_embs, dim=0)


def encode_bundle_regions(
    model: nn.Module,
    bundle_to_image: Dict[str, Path],
    bundle_to_boxes: Optional[Dict[str, List[BoxXYXY]]],
    preprocess_val: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    amp: bool,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[str], torch.Tensor]:
    """Encode all bundle boxes and aggregate to one embedding per bundle."""
    bundle_ids = sorted(bundle_to_image.keys())
    loader = DataLoader(
        BundleRegionDataset(
            bundle_ids=bundle_ids,
            bundle_to_image=bundle_to_image,
            transform=preprocess_val,
            bundle_to_boxes=bundle_to_boxes,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    region_ids, region_embs = encode_images(model, loader, device, amp)
    if region_embs.numel() == 0:
        return [], torch.empty((0, 0), dtype=torch.float32, device=device)

    grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for bundle_id, emb in zip(region_ids, region_embs):
        grouped[bundle_id].append(emb)

    aggregated_ids: List[str] = []
    aggregated_embs: List[torch.Tensor] = []
    for bundle_id in sorted(grouped.keys()):
        stacked = torch.stack(grouped[bundle_id], dim=0)
        mean_emb = F.normalize(stacked.mean(dim=0), p=2, dim=0)
        aggregated_ids.append(bundle_id)
        aggregated_embs.append(mean_emb)

    return aggregated_ids, torch.stack(aggregated_embs, dim=0)


def validate_retrieval(
    model: nn.Module,
    preprocess_val,
    device: torch.device,
    amp: bool,
    val_bundle_to_image: Dict[str, Path],
    val_bundle_to_boxes: Optional[Dict[str, List[BoxXYXY]]],
    val_bundle_to_products: Dict[str, Set[str]],
    product_to_image: Dict[str, Path],
    product_to_text: Dict[str, str],
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    max_val_k: int,
    recall_k: int,
) -> Tuple[float, Dict[str, int]]:
    """Compute Recall@K for bundle->product retrieval."""
    if recall_k > max_val_k:
        raise ValueError("--recall_k must be <= --max_val_k")

    product_ids = sorted(product_to_image.keys())
    product_loader = DataLoader(
        AssetImageDataset(product_ids, product_to_image, preprocess_val, id_to_text=product_to_text, tokenizer=tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    encoded_product_ids, product_embs = encode_images(model, product_loader, device, amp)
    if product_embs.numel() == 0:
        raise RuntimeError("No product embeddings available for validation.")
    pid_to_index = {pid: idx for idx, pid in enumerate(encoded_product_ids)}

    encoded_bundle_ids, bundle_embs = encode_bundle_regions(
        model=model,
        bundle_to_image=val_bundle_to_image,
        bundle_to_boxes=val_bundle_to_boxes,
        preprocess_val=preprocess_val,
        device=device,
        amp=amp,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if bundle_embs.numel() == 0:
        raise RuntimeError("No bundle embeddings available for validation.")

    topk = min(max_val_k, product_embs.shape[0])
    positives_count = Counter()
    recall_sum = 0.0
    recall_count = 0

    gt_index_tensors: Dict[str, torch.Tensor] = {}
    for bundle_id in encoded_bundle_ids:
        gt_pids = val_bundle_to_products.get(bundle_id, set())
        gt_idx = [pid_to_index[pid] for pid in gt_pids if pid in pid_to_index]
        if not gt_idx:
            continue
        gt_tensor = torch.tensor(sorted(set(gt_idx)), dtype=torch.long, device=device)
        gt_index_tensors[bundle_id] = gt_tensor
        positives_count[int(gt_tensor.numel())] += 1

    for start in tqdm(range(0, len(encoded_bundle_ids), batch_size), desc="Val retrieval", leave=False):
        end = min(start + batch_size, len(encoded_bundle_ids))
        batch_ids = encoded_bundle_ids[start:end]
        batch_emb = bundle_embs[start:end]
        sims = batch_emb @ product_embs.T
        _, idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True)

        active_rows: List[int] = []
        row_gt_tensors: List[torch.Tensor] = []
        for row, bundle_id in enumerate(batch_ids):
            gt_tensor = gt_index_tensors.get(bundle_id)
            if gt_tensor is None:
                continue
            active_rows.append(row)
            row_gt_tensors.append(gt_tensor)

        if not active_rows:
            continue

        eval_idx = idx[active_rows, :recall_k]
        lengths = torch.tensor([gt.numel() for gt in row_gt_tensors], dtype=torch.float32, device=device)
        max_gt_len = int(max(lengths).item())
        gt_padded = torch.full(
            (len(row_gt_tensors), max_gt_len),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        for row, gt_tensor in enumerate(row_gt_tensors):
            gt_padded[row, : gt_tensor.numel()] = gt_tensor

        matches = (eval_idx.unsqueeze(-1) == gt_padded.unsqueeze(1)).any(dim=-1)
        hits = matches.sum(dim=1).to(torch.float32)
        recalls = hits / lengths
        recall_sum += float(recalls.sum().item())
        recall_count += int(recalls.numel())

    recall = (recall_sum / recall_count) if recall_count > 0 else 0.0
    return recall, dict(sorted(positives_count.items(), key=lambda x: x[0]))


def get_core_clip_model(image_model: nn.Module) -> nn.Module:
    """Return the underlying OpenCLIP model, with/without DataParallel."""
    if isinstance(image_model, nn.DataParallel):
        return image_model.module.clip_model
    return image_model.clip_model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_metric: float,
    cfg: InditexConfig,
    encoder_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save checkpoint state."""
    payload = {
        "model": model.state_dict(),
        "encoder": encoder_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "args": OmegaConf.to_container(OmegaConf.create(cfg), resolve=True),
    }
    torch.save(payload, path)


def append_metrics(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSONL metrics row."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_gpu_ids(text: str) -> List[int]:
    """Parse comma-separated GPU ids."""
    ids: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    return ids


def train_openclip_retrieval(cfg: InditexConfig, train_manifest: Path, val_manifest: Path, products_manifest: Path, bundles_images_dir: Path, products_images_dir: Path, output_dir: Path) -> None:
    """Train retrieval model with OpenCLIP backend."""
    params = cfg.params

    if params.grad_accum <= 0:
        raise ValueError("params.grad_accum must be >= 1")
    if params.batch_size <= 0:
        raise ValueError("params.batch_size must be > 0")
    if params.epochs <= 0:
        raise ValueError("params.epochs must be > 0")

    set_seed(params.seed)
    device = resolve_device(params.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ensure_dir(output_dir)
    metrics_path = output_dir / "metrics.jsonl"

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    clip_model = clip_model.to(device)
    image_model: nn.Module = OpenCLIPMultimodalEncoder(clip_model).to(device)

    used_gpu_ids: List[int] = []
    if device.type == "cuda" and params.multi_gpu and torch.cuda.device_count() > 1:
        candidate_ids = parse_gpu_ids(params.gpu_ids)
        max_idx = torch.cuda.device_count() - 1
        used_gpu_ids = [gid for gid in candidate_ids if 0 <= gid <= max_idx]
        if len(used_gpu_ids) >= 2:
            image_model = nn.DataParallel(image_model, device_ids=used_gpu_ids)
            print(f"Using DataParallel on GPUs: {used_gpu_ids}")
        else:
            print(
                "Warning: multi_gpu enabled but fewer than 2 valid gpu_ids found. "
                f"Available=0..{max_idx}, requested={candidate_ids}. Using single GPU."
            )
    image_model.train()
    core_model = get_core_clip_model(image_model)

    product_to_image, product_to_text = parse_products_manifest(products_manifest, products_images_dir=products_images_dir)
    train_bundle_to_image, train_bundle_to_products = parse_bundle_manifest(
        train_manifest, product_to_image, bundles_images_dir=bundles_images_dir
    )
    val_bundle_to_image, val_bundle_to_products = parse_bundle_manifest(
        val_manifest, product_to_image, bundles_images_dir=bundles_images_dir
    )
    train_bundle_to_boxes: Optional[Dict[str, List[BoxXYXY]]] = None
    val_bundle_to_boxes: Optional[Dict[str, List[BoxXYXY]]] = None
    if params.use_bundle_boxes:
        all_bundle_to_image = {**train_bundle_to_image, **val_bundle_to_image}
        all_bundle_to_boxes = detect_bundle_boxes_with_cache(
            bundle_to_image=all_bundle_to_image,
            output_dir=output_dir,
            model_id=params.bbox_model_id,
            conf_threshold=params.bbox_conf_threshold,
            iou_threshold=params.bbox_iou_threshold,
            max_boxes_per_image=params.bbox_max_per_image,
            min_area_ratio=params.bbox_min_area_ratio,
            cache_path=params.bbox_cache_path,
        )
        train_bundle_to_boxes = {
            bundle_id: all_bundle_to_boxes.get(bundle_id, [])
            for bundle_id in train_bundle_to_image
        }
        val_bundle_to_boxes = {
            bundle_id: all_bundle_to_boxes.get(bundle_id, [])
            for bundle_id in val_bundle_to_image
        }

    # --- Hard negative mining settings ---
    mine_every = getattr(params, 'mine_every', 3)       # re-mine every N epochs
    hard_neg_top_k = getattr(params, 'hard_neg_top_k', 16)
    max_hard_negatives = getattr(params, 'max_hard_negatives', 4)
    max_positives = getattr(params, 'max_positives', 8)

    hard_negatives: Dict[str, List[str]] = {}  # empty initially — first epoch uses in-batch only

    def build_train_loader(hard_negs: Dict[str, List[str]]) -> Tuple[BundleMultiPositiveDataset, DataLoader]:
        ds = BundleMultiPositiveDataset(
            bundle_to_image=train_bundle_to_image,
            bundle_to_products=train_bundle_to_products,
            product_to_image=product_to_image,
            product_to_text=product_to_text,
            bundle_transform=preprocess_train,
            product_transform=preprocess_train,
            tokenizer=tokenizer,
            max_positives=max_positives,
            hard_negatives=hard_negs,
            max_hard_negatives=max_hard_negatives,
        )
        loader = DataLoader(
            ds,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_multi_positive,
            drop_last=False,
        )
        return ds, loader

    train_dataset, train_loader = build_train_loader(hard_negatives)

    # Get the underlying multimodal encoder (unwrap DataParallel if needed)
    multimodal_encoder = image_model.module if isinstance(image_model, nn.DataParallel) else image_model

    # Optimizer includes: CLIP backbone + fusion_alpha + log_temperature
    param_groups = [
        {"params": core_model.parameters(), "lr": params.lr},
        {"params": [multimodal_encoder.fusion_alpha, multimodal_encoder.log_temperature], "lr": params.lr * 10},
    ]
    optimizer = AdamW(param_groups, weight_decay=params.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / params.grad_accum))
    scheduler = build_scheduler(optimizer, total_steps=params.epochs * updates_per_epoch)
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler(enabled=True) if params.amp and device.type == "cuda" else None
    )

    best_recall = -1.0
    train_boxes = sum(len(v) for v in (train_bundle_to_boxes or {}).values())
    val_boxes = sum(len(v) for v in (val_bundle_to_boxes or {}).values())
    print(
        f"Train bundles: {train_dataset.num_unique_bundles} | "
        f"Train samples (boxes): {len(train_dataset)} | Products indexed: {len(product_to_image)}"
    )
    print(
        f"Use bundle boxes: {bool(params.use_bundle_boxes)} | "
        f"Detected train boxes: {train_boxes} | Detected val boxes: {val_boxes}"
    )
    print(f"Device: {device} | AMP: {bool(scaler is not None)} | multi_gpu={len(used_gpu_ids) >= 2}")
    print(f"Hard negative mining every {mine_every} epochs, top_k={hard_neg_top_k}, max_per_sample={max_hard_negatives}")
    print(f"Multi-positive loss with max_positives={max_positives}, learnable temperature")

    for epoch in range(1, params.epochs + 1):
        epoch_start = time.time()

        # --- Hard negative mining step ---
        if epoch > 1 and (epoch - 1) % mine_every == 0:
            hard_negatives = compute_hard_negatives(
                model=image_model,
                preprocess_val=preprocess_val,
                device=device,
                amp=params.amp,
                bundle_to_image=train_bundle_to_image,
                bundle_to_products=train_bundle_to_products,
                product_to_image=product_to_image,
                product_to_text=product_to_text,
                tokenizer=tokenizer,
                batch_size=params.batch_size,
                num_workers=params.num_workers,
                top_k=hard_neg_top_k,
            )
            train_dataset, train_loader = build_train_loader(hard_negatives)

        image_model.train()
        running_loss = 0.0
        count_steps = 0
        optimizer.zero_grad(set_to_none=True)

        # Get temperature from the model
        temperature = multimodal_encoder.temperature

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{params.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            if batch is None:
                continue
            bundle_imgs = batch["bundle_imgs"].to(device, non_blocking=True)
            product_imgs = batch["product_imgs"].to(device, non_blocking=True)
            pos_mask = batch["pos_mask"].to(device, non_blocking=True)

            if bundle_imgs.shape[0] < 2:
                continue

            with torch.autocast(device_type=device.type, enabled=params.amp and device.type == "cuda"):
                bundle_emb = F.normalize(image_model(bundle_imgs).float(), p=2, dim=1)

                if batch["product_texts"] is not None:
                    product_texts = batch["product_texts"].to(device, non_blocking=True)
                    product_emb = F.normalize(image_model(product_imgs, text=product_texts).float(), p=2, dim=1)
                else:
                    product_emb = F.normalize(image_model(product_imgs).float(), p=2, dim=1)

                # Use learnable temperature (refresh each step)
                temperature = multimodal_encoder.temperature

                # Multi-positive NCE loss
                loss = multi_positive_nce_loss(bundle_emb, product_emb, pos_mask, temperature)
                loss = loss / params.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_update = (step % params.grad_accum == 0) or (step == len(train_loader))
            if do_update:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            loss_item = float(loss.item() * params.grad_accum)
            running_loss += loss_item
            count_steps += 1

            if step % params.log_every == 0:
                avg_loss = running_loss / max(1, count_steps)
                lr_now = optimizer.param_groups[0]["lr"]
                temp_val = float(multimodal_encoder.temperature.item())
                alpha_val = float(torch.sigmoid(multimodal_encoder.fusion_alpha).item())
                print(
                    f"[epoch {epoch} step {step}] loss={avg_loss:.5f} lr={lr_now:.7f} "
                    f"τ={temp_val:.4f} α={alpha_val:.3f} bs={bundle_imgs.shape[0]}"
                )

        train_loss = running_loss / max(1, count_steps)
        recall_val, pos_dist = validate_retrieval(
            model=image_model,
            preprocess_val=preprocess_val,
            device=device,
            amp=params.amp,
            val_bundle_to_image=val_bundle_to_image,
            val_bundle_to_boxes=val_bundle_to_boxes,
            val_bundle_to_products=val_bundle_to_products,
            product_to_image=product_to_image,
            product_to_text=product_to_text,
            tokenizer=tokenizer,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            max_val_k=params.max_val_k,
            recall_k=params.recall_k,
        )
        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]
        temp_val = float(multimodal_encoder.temperature.item())
        alpha_val = float(torch.sigmoid(multimodal_encoder.fusion_alpha).item())

        print(f"Epoch {epoch}: train_loss={train_loss:.6f} recall@{params.recall_k}={recall_val:.6f} τ={temp_val:.4f} α={alpha_val:.3f}")
        print(f"Val #positives per bundle distribution: {pos_dist}")

        metric_row = {
            "epoch": epoch,
            "loss_train": train_loss,
            f"recall@{params.recall_k}": recall_val,
            "lr": lr_now,
            "temperature": temp_val,
            "fusion_alpha": alpha_val,
            "epoch_seconds": epoch_time,
        }
        append_metrics(metrics_path, metric_row)

        if epoch % params.save_every == 0:
            save_checkpoint(
                path=output_dir / f"epoch_{epoch}.pt",
                model=core_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_recall,
                cfg=cfg,
                encoder_state=multimodal_encoder.state_dict(),
            )

        if recall_val > best_recall:
            best_recall = recall_val
            save_checkpoint(
                path=output_dir / "best.pt",
                model=core_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_recall,
                cfg=cfg,
                encoder_state=multimodal_encoder.state_dict(),
            )

    print(f"Training complete. Best recall@{params.recall_k}: {best_recall:.6f}")
    print(f"Final temperature: {float(multimodal_encoder.temperature.item()):.4f}")
    print(f"Final fusion alpha: {float(torch.sigmoid(multimodal_encoder.fusion_alpha).item()):.3f}")
