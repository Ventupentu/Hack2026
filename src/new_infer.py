"""Improved inference: per-crop retrieval, TTA, FAISS, category filtering.

Key improvements over v1 infer.py:
1. Loads fine-tuned OpenCLIP (marqo-fashionSigLIP) checkpoint
2. Per-crop retrieval: each YOLO-detected region gets its own top-K search
3. TTA on product embeddings: multiple augmentations averaged for robustness
4. FAISS index for fast nearest-neighbor search (numpy fallback)
5. Category-aware score boosting using product descriptions

Usage:
    python -m src.infer_v8

    # With fine-tuned checkpoint
    python -m src.infer_v8 infer.checkpoint_path=outputs/retrieval_openclip/best.pt

    # Disable TTA for speed
    python -m src.infer_v8 infer.tta_num_augs=1

    # Lower YOLO threshold to catch more items
    python -m src.infer_v8 params.bbox_conf_threshold=0.15
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import open_clip
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from src.config import InditexConfig
from src.detection import ClothingYOLODetector, ScoredBox, BoxXYXY
from src.utils.metrics import evaluate_bundle_retrieval

try:
    import faiss
    HAS_FAISS = True
except (ImportError, AttributeError, Exception):
    HAS_FAISS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_GROUPS: Dict[str, Set[str]] = {
    "footwear": {
        "SHOES", "SANDAL", "ANKLE BOOT", "BOOT", "FLAT SHOES",
        "FLAT ANKLE BOOT", "FLAT BOOT", "HEELED SHOES",
        "HEELED ANKLE BOOT", "HEELED BOOT", "MOCCASINS",
        "RUNNING SHOES", "SPORT SHOES", "ATHLETIC FOOTWEAR",
        "TRAINERS", "HIGH TOPS", "SPORTY SANDAL", "RAIN BOOT",
        "WEDGE", "VAMP", "PINKY",
    },
    "upper_body": {
        "T-SHIRT", "SHIRT", "SWEATER", "SWEATSHIRT", "BLAZER",
        "CARDIGAN", "POLO SHIRT", "OVERSHIRT", "TOPS AND OTHERS",
        "KNITTED WAISTCOAT", "WAISTCOAT", "BODYSUIT",
    },
    "outerwear": {
        "COAT", "WIND-JACKET", "ANORAK", "TRENCH-RAINCOAT",
        "4 COAT", "SLEEVELESS PAD. JACKET",
    },
    "lower_body": {
        "TROUSERS", "BERMUDA", "SKIRT", "LEGGINGS", "SHORTS",
    },
    "full_body": {
        "DRESS", "OVERALL", "BIB OVERALL", "PYJAMAS", "NIGHTIE", "SWIMSUIT",
    },
    "accessories": {
        "HAT", "BELT", "GLASSES", "SCARF", "FOULARD", "SHAWL",
        "TIE", "GLOVES", "HAND BAG-RUCKSACK", "PURSE-WALLET",
        "WALLETS", "ACCESSORIES", "IMIT-JEWELLER",
    },
    "underwear": {
        "BRA", "PANTY", "SOCKS", "UNDERPANT", "UNDERWEAR",
        "STOCKINGS-TIGHTS",
    },
    "baby": {
        "BABY T-SHIRT", "BABY TROUSERS", "BABY SWEATER", "BABY SHIRT",
        "BABY JACKET", "BABY DRESS", "BABY BERMUDAS", "BABY SKIRT",
        "BABY CARDIGAN", "BABY LEGGINGS", "BABY SOCKS", "BABY BONNET",
        "BABY WAISTCOAT", "BABY TRACKSUIT", "BABY WIND JACKET",
        "LEISURE AND SPORTS",
    },
}

# Flat lookup: description -> category
_DESC_TO_CATEGORY: Dict[str, str] = {}
for _cat, _descs in CATEGORY_GROUPS.items():
    for _d in _descs:
        _DESC_TO_CATEGORY[_d] = _cat


# ---------------------------------------------------------------------------
# Helpers (reused from retrieval_openclip)
# ---------------------------------------------------------------------------

_GENDER_UNKNOWN = 0  # section id for unknown / unisex


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def open_image_safe(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as exc:
        print(f"Warning: cannot read {path} ({exc})")
        return None


def crop_with_box(image: Image.Image, box: BoxXYXY) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = image.size
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(1, min(x2, w)), max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


def build_image_map(image_dir: Path) -> Dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return {p.stem: p for p in image_dir.iterdir() if p.is_file()}


def parse_ks(text: str) -> List[int]:
    return sorted({int(t.strip()) for t in text.split(",") if t.strip()})


def extract_ts_from_url(url: object) -> Optional[int]:
    """Extract URL query parameter ``ts`` as integer timestamp."""
    if pd.isna(url):
        return None
    text = str(url).strip()
    if not text:
        return None
    query = parse_qs(urlparse(text).query)
    values = query.get("ts")
    if not values:
        return None
    value = str(values[0]).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _ts_to_seconds(ts_value: int) -> float:
    """Convert ts to seconds (supports seconds or milliseconds)."""
    return float(ts_value) / 1000.0 if ts_value >= 10**12 else float(ts_value)


def _ts_period_keys(ts_value: int) -> Tuple[str, str]:
    """Return (month_key, quarter_key) in UTC, e.g. ('2025-09', '2025-Q3')."""
    if ts_value >= 10**12:
        ts = pd.Timestamp(ts_value, unit="ms", tz="UTC")
    else:
        ts = pd.Timestamp(ts_value, unit="s", tz="UTC")
    month_key = f"{ts.year:04d}-{ts.month:02d}"
    quarter_key = f"{ts.year:04d}-Q{((ts.month - 1) // 3) + 1}"
    return month_key, quarter_key


def load_asset_timestamps(df: pd.DataFrame, id_col: str, url_col: str) -> Dict[str, int]:
    """Build id -> ts map from a dataframe with URL column."""
    out: Dict[str, int] = {}
    for row in df.itertuples(index=False):
        asset_id = str(getattr(row, id_col))
        url = getattr(row, url_col, None)
        ts_value = extract_ts_from_url(url)
        if ts_value is not None:
            out[asset_id] = ts_value
    return out


def compute_ts_adjustment(
    bundle_ts: Optional[int],
    product_ts: Optional[int],
    *,
    delta_weight: float,
    decay_hours: float,
    bonus_same_date: float,
    bonus_same_month: float,
    bonus_same_quarter: float,
    penalty_diff_quarter: float,
) -> float:
    """Compute additive score adjustment based on timestamp proximity."""
    if bundle_ts is None or product_ts is None:
        return 0.0

    b_sec = _ts_to_seconds(bundle_ts)
    p_sec = _ts_to_seconds(product_ts)
    delta_hours = abs(b_sec - p_sec) / 3600.0

    # Continuous proximity term in [-delta_weight, +delta_weight].
    safe_decay = max(decay_hours, 1e-6)
    proximity = math.exp(-delta_hours / safe_decay)
    adjustment = float(delta_weight) * ((2.0 * proximity) - 1.0)

    # Discrete period prior.
    b_ts_int = int(round(bundle_ts))
    p_ts_int = int(round(product_ts))
    if b_ts_int >= 10**12:
        b_date = pd.Timestamp(b_ts_int, unit="ms", tz="UTC").date().isoformat()
    else:
        b_date = pd.Timestamp(b_ts_int, unit="s", tz="UTC").date().isoformat()
    if p_ts_int >= 10**12:
        p_date = pd.Timestamp(p_ts_int, unit="ms", tz="UTC").date().isoformat()
    else:
        p_date = pd.Timestamp(p_ts_int, unit="s", tz="UTC").date().isoformat()

    if b_date == p_date:
        adjustment += float(bonus_same_date)
    else:
        b_month, b_quarter = _ts_period_keys(b_ts_int)
        p_month, p_quarter = _ts_period_keys(p_ts_int)
        if b_month == p_month:
            adjustment += float(bonus_same_month)
        elif b_quarter == p_quarter:
            adjustment += float(bonus_same_quarter)
        else:
            adjustment += float(penalty_diff_quarter)
    return adjustment


# ---------------------------------------------------------------------------
# Gender & category helpers
# ---------------------------------------------------------------------------

def load_bundle_genders(bundles_df: pd.DataFrame) -> Dict[str, int]:
    """Return bundle_id -> section (int) from bundles dataframe."""
    out: Dict[str, int] = {}
    for row in bundles_df.itertuples(index=False):
        bid = str(row.bundle_asset_id)
        section = row.bundle_id_section
        if not pd.isna(section):
            out[bid] = int(section)
    return out


def load_product_genders(products_gender_csv: Path) -> Dict[str, int]:
    """Return product_id -> gender (int) from product_dataset_with_gender.csv."""
    if not products_gender_csv.exists():
        return {}
    df = pd.read_csv(products_gender_csv)
    if "gender" not in df.columns:
        return {}
    out: Dict[str, int] = {}
    for row in df.itertuples(index=False):
        pid = str(row.product_asset_id)
        gender = row.gender
        out[pid] = int(gender) if not pd.isna(gender) else _GENDER_UNKNOWN
    return out


def load_product_categories(products_csv: Path) -> Dict[str, str]:
    """Return product_id -> product_description (uppercase) from product CSV."""
    df = pd.read_csv(products_csv)
    out: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        pid = str(row.product_asset_id)
        desc = str(row.product_description) if not pd.isna(row.product_description) else ""
        out[pid] = desc.strip().upper()
    return out


def filter_cross_gender(
    scored_products: List[Tuple[str, float]],
    bundle_gender: int,
    product_to_gender: Dict[str, int],
) -> List[Tuple[str, float]]:
    """Remove products whose *known* gender differs from the bundle's known gender.

    - If the bundle gender is unknown (0) → no filtering.
    - Products with unknown gender (0) → always kept.
    - Products with same gender → kept.
    - Products with different known gender → dropped.
    """
    if bundle_gender == _GENDER_UNKNOWN:
        return scored_products
    return [
        (pid, score)
        for pid, score in scored_products
        if product_to_gender.get(pid, _GENDER_UNKNOWN) in (_GENDER_UNKNOWN, bundle_gender)
    ]


def deduplicate_by_category(
    scored_products: List[Tuple[str, float]],
    product_to_category: Dict[str, str],
    max_per_category: int,
) -> List[Tuple[str, float]]:
    """Keep at most ``max_per_category`` products per product_description.

    Input must be sorted by score descending.  Products with empty/unknown
    category are always kept (no limit).
    """
    if max_per_category <= 0:
        return scored_products
    category_counts: Dict[str, int] = defaultdict(int)
    result: List[Tuple[str, float]] = []
    for pid, score in scored_products:
        cat = product_to_category.get(pid, "")
        if cat and category_counts[cat] >= max_per_category:
            continue
        if cat:
            category_counts[cat] += 1
        result.append((pid, score))
    return result


def apply_score_threshold(
    scored_products: List[Tuple[str, float]],
    threshold: float,
) -> List[Tuple[str, float]]:
    """Drop products below a cosine similarity threshold."""
    if threshold <= 0.0:
        return scored_products
    return [(pid, score) for pid, score in scored_products if score >= threshold]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class MultimodalEncoder(torch.nn.Module):
    """Wrapper: image-only for bundles, image+text for products.

    Uses the same learned fusion_alpha as training:
        alpha * image_features + (1 - alpha) * text_features
    """

    def __init__(self, clip_model: torch.nn.Module, fusion_alpha: float = 0.5) -> None:
        super().__init__()
        self.clip_model = clip_model
        self._fusion_alpha = fusion_alpha  # already sigmoid-ed

    def forward(
        self, images: torch.Tensor, text: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        img_feats = self.clip_model.encode_image(images)
        if text is not None:
            txt_feats = self.clip_model.encode_text(text)
            a = self._fusion_alpha
            return a * img_feats + (1.0 - a) * txt_feats
        return img_feats


def load_openclip_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[MultimodalEncoder, Any, Any]:
    """Load marqo-fashionSigLIP, optionally from a fine-tuned checkpoint."""
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    fusion_alpha = 0.5  # default: equal weight

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            clip_model.load_state_dict(ckpt["model"])
            # Load learned fusion gate from encoder state
            encoder_state = ckpt.get("encoder", {})
            if encoder_state and "fusion_alpha" in encoder_state:
                raw_alpha = encoder_state["fusion_alpha"]
                if hasattr(raw_alpha, 'item'):
                    raw_alpha = raw_alpha.item()
                fusion_alpha = float(torch.sigmoid(torch.tensor(raw_alpha)).item())
                print(f"  Loaded learned fusion_alpha: {fusion_alpha:.4f} (raw={raw_alpha:.4f})")
            epoch = ckpt.get("epoch", "?")
            metric = ckpt.get("best_metric", "?")
            print(f"Loaded checkpoint: {ckpt_path} (epoch={epoch}, recall={metric})")
        else:
            print(f"Warning: checkpoint not found at {ckpt_path}, using pretrained weights.")

    print(f"  Fusion: {fusion_alpha:.1%} image + {1-fusion_alpha:.1%} text")
    encoder = MultimodalEncoder(clip_model, fusion_alpha=fusion_alpha).to(device).eval()
    return encoder, preprocess_val, tokenizer


# ---------------------------------------------------------------------------
# TTA (Test-Time Augmentation) transforms
# ---------------------------------------------------------------------------

def _build_tta_pil_transforms(n: int) -> List[Callable[[Image.Image], Image.Image]]:
    """Return up to n PIL-level augmentations applied *before* model preprocessing."""

    def identity(img: Image.Image) -> Image.Image:
        return img

    def hflip(img: Image.Image) -> Image.Image:
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def center_crop_85(img: Image.Image) -> Image.Image:
        w, h = img.size
        nw, nh = int(w * 0.85), int(h * 0.85)
        left, top = (w - nw) // 2, (h - nh) // 2
        return img.crop((left, top, left + nw, top + nh))

    def top_crop(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((0, 0, w, int(h * 0.8)))

    def bottom_crop(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((0, int(h * 0.2), w, h))

    pool = [identity, hflip, center_crop_85, top_crop, bottom_crop]
    return pool[: max(1, min(n, len(pool)))]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ProductTTADataset(Dataset):
    """Product images × TTA transforms, with optional text."""

    def __init__(
        self,
        product_ids: Sequence[str],
        product_to_image: Dict[str, Path],
        product_to_text: Dict[str, str],
        preprocess: Callable,
        tokenizer: Any,
        tta_fns: List[Callable],
    ) -> None:
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.items: List[Tuple[str, Path, str, Callable]] = []
        for pid in product_ids:
            path = product_to_image.get(pid)
            if path is None or not path.exists():
                continue
            text = product_to_text.get(pid, "")
            for fn in tta_fns:
                self.items.append((pid, path, text, fn))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        pid, path, text, tta_fn = self.items[idx]
        img = open_image_safe(path)
        if img is None:
            return None
        img = tta_fn(img)
        out: Dict[str, Any] = {"id": pid, "img": self.preprocess(img)}
        if text and self.tokenizer:
            out["text"] = self.tokenizer(text).squeeze(0)
        return out


class BundleCropDataset(Dataset):
    """One item per detected crop (or full-image fallback) per bundle.

    Supports TTA: generates original + horizontally flipped versions.
    """

    def __init__(
        self,
        crop_items: List[Tuple[str, Path, Optional[BoxXYXY], float]],
        preprocess: Callable,
        use_hflip_tta: bool = True,
    ) -> None:
        self.preprocess = preprocess
        # Expand items with optional hflip TTA
        self.items: List[Tuple[str, Path, Optional[BoxXYXY], float, bool]] = []
        for bid, path, box, conf in crop_items:
            self.items.append((bid, path, box, conf, False))  # original
            if use_hflip_tta:
                self.items.append((bid, path, box, conf, True))  # hflip

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        bundle_id, img_path, box, confidence, do_flip = self.items[idx]
        img = open_image_safe(img_path)
        if img is None:
            return None
        if box is not None:
            img = crop_with_box(img, box)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return {
            "bundle_id": bundle_id,
            "confidence": confidence,
            "img": self.preprocess(img),
        }


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for key in keys:
        vals = [item[key] for item in batch]
        if torch.is_tensor(vals[0]):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _encode_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    desc: str = "Encoding",
) -> Tuple[List[str], np.ndarray, List[float]]:
    """Encode a DataLoader returning (ids, embeddings, confidences)."""
    all_ids: List[str] = []
    all_embs: List[np.ndarray] = []
    all_confs: List[float] = []

    for batch in tqdm(loader, leave=False, desc=desc):
        if batch is None:
            continue
        imgs = batch["img"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            if "text" in batch:
                texts = batch["text"].to(device, non_blocking=True)
                feats = model(imgs, text=texts)
            else:
                feats = model(imgs)
        feats = F.normalize(feats.float(), p=2, dim=1)
        all_embs.append(feats.cpu().numpy())
        all_ids.extend(batch.get("id", batch.get("bundle_id")))
        if "confidence" in batch:
            all_confs.extend(
                [float(c) for c in batch["confidence"]]
            )

    if not all_embs:
        return [], np.zeros((0, 0), dtype=np.float32), []
    return all_ids, np.concatenate(all_embs, axis=0), all_confs


def encode_products(
    product_ids: List[str],
    product_to_image: Dict[str, Path],
    product_to_text: Dict[str, str],
    model: torch.nn.Module,
    preprocess: Callable,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    n_tta: int,
) -> Tuple[List[str], np.ndarray]:
    """Encode all products, optionally with TTA averaging."""
    tta_fns = _build_tta_pil_transforms(n_tta)
    print(f"  TTA augmentations: {len(tta_fns)} per product")

    dataset = ProductTTADataset(
        product_ids, product_to_image, product_to_text,
        preprocess, tokenizer, tta_fns,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    raw_ids, raw_embs, _ = _encode_loader(model, loader, device, amp, "Encoding products")

    if len(tta_fns) <= 1:
        return raw_ids, raw_embs

    # Aggregate TTA: average embeddings per product, then re-normalize
    pid_to_embs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for pid, emb in zip(raw_ids, raw_embs):
        pid_to_embs[pid].append(emb)

    final_pids = sorted(pid_to_embs.keys())
    stacked = np.stack([np.mean(pid_to_embs[p], axis=0) for p in final_pids])
    norms = np.linalg.norm(stacked, axis=1, keepdims=True).clip(min=1e-8)
    return final_pids, (stacked / norms).astype(np.float32)


def encode_bundle_crops(
    crop_items: List[Tuple[str, Path, Optional[BoxXYXY], float]],
    model: torch.nn.Module,
    preprocess: Callable,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    """Encode bundle crops. Returns bundle_id -> [(embedding, confidence), ...]."""
    dataset = BundleCropDataset(crop_items, preprocess)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    ids, embs, confs = _encode_loader(model, loader, device, amp, "Encoding bundle crops")

    grouped: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
    for bid, emb, conf in zip(ids, embs, confs):
        grouped[bid].append((emb, conf))
    return grouped


# ---------------------------------------------------------------------------
# Search index
# ---------------------------------------------------------------------------

class ProductIndex:
    """Fast nearest-neighbor search over product embeddings."""

    def __init__(self, product_ids: List[str], embeddings: np.ndarray) -> None:
        self.product_ids = list(product_ids)
        self.embeddings = embeddings.astype(np.float32)
        self._faiss_index = None
        if HAS_FAISS and embeddings.shape[0] > 0:
            self._faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self._faiss_index.add(self.embeddings)
            print(f"  FAISS index built ({embeddings.shape[0]} vectors, dim={embeddings.shape[1]})")
        else:
            print(f"  Using numpy fallback for search ({embeddings.shape})")

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k for a BATCH of queries. Returns (indices, scores) [Q, k]."""
        k = min(k, len(self.product_ids))
        q = query.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, k)
            return indices, scores
        # Numpy fallback
        sims = q @ self.embeddings.T
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        rows = np.arange(q.shape[0])[:, None]
        part_scores = sims[rows, part]
        order = np.argsort(-part_scores, axis=1)
        return np.take_along_axis(part, order, axis=1), np.take_along_axis(part_scores, order, axis=1)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_bundles(
    bundle_ids: List[str],
    bundle_image_map: Dict[str, Path],
    params,
) -> Dict[str, List[ScoredBox]]:
    """Detect clothing boxes for all bundles (with confidence scores)."""
    try:
        detector = ClothingYOLODetector(
            model_id=params.bbox_model_id,
            conf_threshold=params.bbox_conf_threshold,
            iou_threshold=params.bbox_iou_threshold,
            max_boxes_per_image=params.bbox_max_per_image,
            min_area_ratio=params.bbox_min_area_ratio,
        )
    except ModuleNotFoundError:
        print("Warning: ultralyticsplus not available, using full-image fallback.")
        return {bid: [] for bid in bundle_ids}

    result: Dict[str, List[ScoredBox]] = {}
    for bid in tqdm(bundle_ids, desc="Detecting clothing"):
        path = bundle_image_map.get(bid)
        if path is None or not path.exists():
            result[bid] = []
            continue
        result[bid] = detector.detect_boxes(path)
    return result


def build_crop_items(
    bundle_ids: List[str],
    bundle_image_map: Dict[str, Path],
    bundle_to_boxes: Dict[str, List[ScoredBox]],
) -> List[Tuple[str, Path, Optional[BoxXYXY], float]]:
    """Build flat list of (bundle_id, path, box_or_None, confidence)."""
    items: List[Tuple[str, Path, Optional[BoxXYXY], float]] = []
    box_pad = 0.15  # 15% padding around detected boxes
    for bid in bundle_ids:
        path = bundle_image_map.get(bid)
        if path is None or not path.exists():
            continue
        boxes = bundle_to_boxes.get(bid, [])
        if boxes:
            for x1, y1, x2, y2, conf in boxes:
                # Pad bounding box by 15% for better context
                w, h = x2 - x1, y2 - y1
                px, py = int(w * box_pad), int(h * box_pad)
                padded_box = (x1 - px, y1 - py, x2 + px, y2 + py)
                items.append((bid, path, padded_box, conf))
            # Also add full image with moderate confidence
            items.append((bid, path, None, 0.6))
        else:
            # No detections -> use full image
            items.append((bid, path, None, 1.0))
    return items


# ---------------------------------------------------------------------------
# Per-crop retrieval + post-processing
# ---------------------------------------------------------------------------


def retrieve_per_crop(
    bundle_crops: Dict[str, List[Tuple[np.ndarray, float]]],
    product_index: ProductIndex,
    product_categories: Dict[str, str],
    top_k_per_crop: int,
    top_n_submit: int,
    bundle_to_gender: Optional[Dict[str, int]] = None,
    product_to_gender: Optional[Dict[str, int]] = None,
    gender_filter: bool = True,
    max_per_category: int = 2,
    score_threshold: float = 0.0,
    bundle_to_ts: Optional[Dict[str, int]] = None,
    product_to_ts: Optional[Dict[str, int]] = None,
    ts_rerank_enabled: bool = False,
    ts_delta_weight: float = 0.0,
    ts_decay_hours: float = 720.0,
    ts_bonus_same_date: float = 0.0,
    ts_bonus_same_month: float = 0.0,
    ts_bonus_same_quarter: float = 0.0,
    ts_penalty_diff_quarter: float = 0.0,
) -> Dict[str, List[str]]:
    """Per-crop retrieval with score merging + hard post-processing pipeline.

    Pipeline per bundle (all steps on scored list, sorted desc):
      1. Score threshold  — drop low-confidence results
      2. Gender filter    — discard known cross-gender products
      3. Category dedup   — max N products per product_description
      4. TS rerank        — additive bonus/penalty by timestamp delta
      5. Top-K            — take final top_n_submit (may return fewer)
    """
    results: Dict[str, List[str]] = {}

    for bundle_id, crops in tqdm(bundle_crops.items(), desc="Per-crop retrieval", leave=False):
        candidates: Dict[str, float] = {}

        for emb, conf in crops:
            indices, scores = product_index.search(emb, top_k_per_crop)
            for idx, sim in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                pid = product_index.product_ids[int(idx)]
                combined = float(sim) * max(float(conf), 0.1)
                # SUM aggregation: products matching multiple crops score higher
                candidates[pid] = candidates.get(pid, 0.0) + combined

        # Sort by score descending — base ranking for post-processing
        ranked: List[Tuple[str, float]] = sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        )

        # 1) Score threshold: drop low-confidence results
        ranked = apply_score_threshold(ranked, score_threshold)

        # 2) Gender filter: discard products with a different *known* gender
        if gender_filter and bundle_to_gender and product_to_gender:
            bundle_gender = bundle_to_gender.get(bundle_id, _GENDER_UNKNOWN)
            ranked = filter_cross_gender(ranked, bundle_gender, product_to_gender)

        # 3) Category diversity: max N products per description category
        if max_per_category > 0 and product_categories:
            ranked = deduplicate_by_category(ranked, product_categories, max_per_category)

        # 4) Timestamp rerank: score += f(Δts) + period prior bonuses/penalties.
        if ts_rerank_enabled and bundle_to_ts and product_to_ts:
            bundle_ts = bundle_to_ts.get(bundle_id)
            reranked: List[Tuple[str, float]] = []
            for pid, base_score in ranked:
                adjustment = compute_ts_adjustment(
                    bundle_ts=bundle_ts,
                    product_ts=product_to_ts.get(pid),
                    delta_weight=ts_delta_weight,
                    decay_hours=ts_decay_hours,
                    bonus_same_date=ts_bonus_same_date,
                    bonus_same_month=ts_bonus_same_month,
                    bonus_same_quarter=ts_bonus_same_quarter,
                    penalty_diff_quarter=ts_penalty_diff_quarter,
                )
                reranked.append((pid, base_score + adjustment))
            ranked = sorted(reranked, key=lambda x: x[1], reverse=True)

        # Take top final (may be < top_n_submit — that's OK)
        results[bundle_id] = [pid for pid, _ in ranked[:top_n_submit]]

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
    """Compute validation metrics using per-crop predictions."""
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
    top_n_submit = min(cfg.infer.top_n_submit, 15)
    checkpoint_path = cfg.infer.checkpoint_path
    n_tta = cfg.infer.tta_num_augs
    per_crop_topk = cfg.infer.per_crop_topk

    # Post-processing params from config
    gender_filter = bool(getattr(cfg.infer, "gender_filter", True))
    max_per_category = int(getattr(cfg.infer, "max_per_category", 2))
    score_threshold = float(getattr(cfg.infer, "score_threshold", 0.0))
    ts_rerank_enabled = bool(getattr(cfg.infer, "ts_rerank_enabled", False))
    ts_delta_weight = float(getattr(cfg.infer, "ts_delta_weight", 0.0))
    ts_decay_hours = float(getattr(cfg.infer, "ts_decay_hours", 720.0))
    ts_bonus_same_date = float(getattr(cfg.infer, "ts_bonus_same_date", 0.0))
    ts_bonus_same_month = float(getattr(cfg.infer, "ts_bonus_same_month", 0.0))
    ts_bonus_same_quarter = float(getattr(cfg.infer, "ts_bonus_same_quarter", 0.0))
    ts_penalty_diff_quarter = float(getattr(cfg.infer, "ts_penalty_diff_quarter", 0.0))

    set_seed(seed)

    # ---- Load model ----
    print("=" * 60)
    print("INFER v8 — Per-crop retrieval + TTA + Post-processing")
    print("=" * 60)
    model, preprocess, tokenizer = load_openclip_model(checkpoint_path, device)
    print(f"Device: {device} | AMP: {amp} | TTA: {n_tta} | TopK/crop: {per_crop_topk}")

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
    bundle_to_ts = load_asset_timestamps(bundles_df, "bundle_asset_id", "bundle_image_url")
    product_to_ts = load_asset_timestamps(products_df, "product_asset_id", "product_image_url")
    # Build raw description category map for dedup (uppercase description)
    product_to_category = load_product_categories(products_csv)
    print(f"Products: {len(product_ids)} with images / {len(all_product_ids)} total")

    # ---- Load gender maps ----
    bundle_to_gender = load_bundle_genders(bundles_df)
    products_gender_csv = data_dir / "product_dataset_with_gender.csv"
    product_to_gender = load_product_genders(products_gender_csv)

    if gender_filter and bundle_to_gender and product_to_gender:
        print(f"Gender filter enabled: {len(bundle_to_gender)} bundles, {len(product_to_gender)} products")
    else:
        print("Gender filter disabled.")
    print(f"Category dedup: max_per_category={max_per_category} | Score threshold: {score_threshold}")
    if ts_rerank_enabled:
        print(
            "TS rerank enabled: "
            f"delta_weight={ts_delta_weight}, decay_hours={ts_decay_hours}, "
            f"bonus_day={ts_bonus_same_date}, bonus_month={ts_bonus_same_month}, "
            f"bonus_quarter={ts_bonus_same_quarter}, penalty_out_quarter={ts_penalty_diff_quarter}"
        )
    else:
        print("TS rerank disabled.")

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
        val_crop_items = build_crop_items(val_bundle_ids, bundle_image_map, val_boxes)
        val_crop_embs = encode_bundle_crops(
            val_crop_items, model, preprocess, device, batch_size, num_workers, amp,
        )
        val_predictions = retrieve_per_crop(
            val_crop_embs, search_index, product_to_category,
            per_crop_topk, top_n_submit,
            bundle_to_gender=bundle_to_gender,
            product_to_gender=product_to_gender,
            gender_filter=gender_filter,
            max_per_category=max_per_category,
            score_threshold=score_threshold,
            bundle_to_ts=bundle_to_ts,
            product_to_ts=product_to_ts,
            ts_rerank_enabled=ts_rerank_enabled,
            ts_delta_weight=ts_delta_weight,
            ts_decay_hours=ts_decay_hours,
            ts_bonus_same_date=ts_bonus_same_date,
            ts_bonus_same_month=ts_bonus_same_month,
            ts_bonus_same_quarter=ts_bonus_same_quarter,
            ts_penalty_diff_quarter=ts_penalty_diff_quarter,
        )

        val_metrics = validate(val_bundle_ids, val_predictions, gt_map, eval_ks)
        print("  Validation metrics:")
        for key, value in val_metrics.items():
            print(f"    {key}: {value:.6f}")

        metrics_out = output_dir / "val_metrics_v8.json"
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

    test_crop_items = build_crop_items(test_bundle_ids, bundle_image_map, test_boxes)
    test_crop_embs = encode_bundle_crops(
        test_crop_items, model, preprocess, device, batch_size, num_workers, amp,
    )
    test_predictions = retrieve_per_crop(
        test_crop_embs, search_index, product_to_category,
        per_crop_topk, top_n_submit,
        bundle_to_gender=bundle_to_gender,
        product_to_gender=product_to_gender,
        gender_filter=gender_filter,
        max_per_category=max_per_category,
        score_threshold=score_threshold,
        bundle_to_ts=bundle_to_ts,
        product_to_ts=product_to_ts,
        ts_rerank_enabled=ts_rerank_enabled,
        ts_delta_weight=ts_delta_weight,
        ts_decay_hours=ts_decay_hours,
        ts_bonus_same_date=ts_bonus_same_date,
        ts_bonus_same_month=ts_bonus_same_month,
        ts_bonus_same_quarter=ts_bonus_same_quarter,
        ts_penalty_diff_quarter=ts_penalty_diff_quarter,
    )

    submission_rows: List[Dict[str, str]] = []
    for bid in test_bundle_ids:
        preds = test_predictions.get(bid, [])[:top_n_submit]
        for pid in preds:
            submission_rows.append({"bundle_asset_id": bid, "product_asset_id": pid})

    submission_df = pd.DataFrame(submission_rows, columns=["bundle_asset_id", "product_asset_id"])
    submission_out = output_dir / "test_submission_v8.csv"
    submission_df.to_csv(submission_out, index=False)

    print(f"\n{'=' * 60}")
    print(f"Submission saved: {submission_out} ({len(submission_df)} rows)")
    print(f"Bundles: {len(test_bundle_ids)} | Avg products/bundle: {len(submission_df)/len(test_bundle_ids):.1f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
