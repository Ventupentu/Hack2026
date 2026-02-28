"""Improved pretrained retrieval baseline for bundle -> products prediction."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise ModuleNotFoundError(
        "Missing PyTorch dependencies. Install with:\n"
        "  pip install torch torchvision"
    ) from exc

from src.models.feature_extractors import build_pretrained_encoder, resolve_device
from src.utils.metrics import evaluate_bundle_retrieval


class AssetImageDataset(Dataset):
    """Image dataset keyed by asset id."""

    def __init__(self, asset_ids: Sequence[str], image_map: Dict[str, Path], transform):
        self.asset_ids = list(asset_ids)
        self.image_map = image_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.asset_ids)

    def __getitem__(self, idx: int):
        asset_id = self.asset_ids[idx]
        image_path = self.image_map[asset_id]
        image = Image.open(image_path).convert("RGB")
        return asset_id, self.transform(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved pretrained visual retrieval baseline")
    parser.add_argument("--bundles-csv", type=Path, default=Path("data/bundles_dataset.csv"))
    parser.add_argument("--products-csv", type=Path, default=Path("data/product_dataset.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/bundles_product_match_train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/bundles_product_match_test.csv"))
    parser.add_argument("--bundle-images-dir", type=Path, default=Path("data/bundle_images"))
    parser.add_argument("--product-images-dir", type=Path, default=Path("data/product_images"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/test_submission.csv"))
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/val_metrics.json"))
    parser.add_argument(
        "--model-name",
        type=str,
        default="fashionclip",
        choices=["resnet18", "resnet50", "efficientnet_b0", "fashionclip", "clip", "hf_clip"],
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="",
        help=(
            "Optional HuggingFace model id for CLIP-like encoders. "
            "If empty and model-name=fashionclip, uses patrickjohncyh/fashion-clip."
        ),
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated GPU ids for DataParallel, e.g. '0,1'. Empty -> all visible GPUs.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ks", type=str, default="1,5,10,15")
    parser.add_argument("--top-n-submit", type=int, default=15)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument(
        "--bundle-view-mode",
        type=str,
        default="full+5crop",
        choices=["full", "full+5crop", "full+grid2x2"],
        help="Views used for each bundle image.",
    )
    parser.add_argument(
        "--score-agg",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="How to aggregate scores from multiple bundle views.",
    )
    parser.add_argument(
        "--product-tta-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use horizontal flip test-time augmentation for product embeddings.",
    )
    parser.add_argument(
        "--use-section-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter product candidates by section->description prior from train data.",
    )
    parser.add_argument(
        "--top-descriptions-per-section",
        type=int,
        default=30,
        help="How many product descriptions to keep per section for candidate filtering.",
    )
    return parser.parse_args()


def parse_ks(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("--eval-ks must contain at least one positive integer.")
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


def parse_gpu_ids(gpu_ids: str) -> List[int]:
    if not gpu_ids.strip():
        return []
    out = []
    for token in gpu_ids.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def maybe_wrap_dataparallel(
    model: nn.Module,
    device: torch.device,
    gpu_ids: Sequence[int],
) -> Tuple[nn.Module, torch.device, List[int]]:
    """Wrap model with DataParallel when 2+ GPUs are requested/available."""
    if device.type != "cuda":
        return model, device, []

    available = torch.cuda.device_count()
    if available <= 1:
        return model, torch.device("cuda:0"), [0]

    chosen = list(gpu_ids) if gpu_ids else list(range(available))
    chosen = [gpu for gpu in chosen if 0 <= gpu < available]
    if not chosen:
        chosen = [0]

    primary = chosen[0]
    runtime_device = torch.device(f"cuda:{primary}")
    model = model.to(runtime_device)
    if len(chosen) > 1:
        model = nn.DataParallel(model, device_ids=chosen)
    return model, runtime_device, chosen


@torch.inference_mode()
def encode_assets(
    asset_ids: Sequence[str],
    image_map: Dict[str, Path],
    model: nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    embedding_dim: int,
    use_amp: bool,
    tta_flip: bool = False,
) -> Tuple[List[str], np.ndarray]:
    ids = [asset_id for asset_id in asset_ids if asset_id in image_map]
    if not ids:
        return [], np.zeros((0, embedding_dim), dtype=np.float32)

    dataset = AssetImageDataset(ids, image_map=image_map, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    output_ids: List[str] = []
    output_embeddings: List[np.ndarray] = []
    amp_enabled = use_amp and device.type == "cuda"

    for batch_ids, batch_images in tqdm(loader, desc=f"Encoding {len(dataset)} images", leave=False):
        output_ids.extend(list(batch_ids))
        batch_images = batch_images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            features = model(batch_images)
            if tta_flip:
                flipped = torch.flip(batch_images, dims=[3])
                features_flip = model(flipped)
                features = 0.5 * (features + features_flip)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        features = F.normalize(features, p=2, dim=1)
        output_embeddings.append(features.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(output_embeddings, axis=0)
    return output_ids, embeddings


def generate_bundle_views(image: Image.Image, mode: str) -> List[Image.Image]:
    """Create multiple crops for a bundle image."""
    image = image.convert("RGB")
    if mode == "full":
        return [image]

    w, h = image.size
    views = [image]

    if mode == "full+grid2x2":
        mid_x = w // 2
        mid_y = h // 2
        boxes = [
            (0, 0, mid_x, mid_y),
            (mid_x, 0, w, mid_y),
            (0, mid_y, mid_x, h),
            (mid_x, mid_y, w, h),
        ]
    else:  # full+5crop
        crop_w = max(16, int(w * 0.75))
        crop_h = max(16, int(h * 0.75))
        boxes = [
            (0, 0, crop_w, crop_h),  # top-left
            (w - crop_w, 0, w, crop_h),  # top-right
            (0, h - crop_h, crop_w, h),  # bottom-left
            (w - crop_w, h - crop_h, w, h),  # bottom-right
            ((w - crop_w) // 2, (h - crop_h) // 2, (w + crop_w) // 2, (h + crop_h) // 2),  # center
        ]

    for left, top, right, bottom in boxes:
        left = max(0, min(left, w - 1))
        top = max(0, min(top, h - 1))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))
        views.append(image.crop((left, top, right, bottom)))
    return views


@torch.inference_mode()
def encode_bundles_multiview(
    bundle_ids: Sequence[str],
    image_map: Dict[str, Path],
    model: nn.Module,
    transform,
    device: torch.device,
    embedding_dim: int,
    use_amp: bool,
    view_mode: str,
) -> Tuple[List[str], List[np.ndarray]]:
    """Encode each bundle with multiple views, returning list of [num_views, dim]."""
    ids = [bundle_id for bundle_id in bundle_ids if bundle_id in image_map]
    if not ids:
        return [], []

    amp_enabled = use_amp and device.type == "cuda"
    out_ids: List[str] = []
    out_embeddings: List[np.ndarray] = []

    for bundle_id in tqdm(ids, desc=f"Encoding {len(ids)} bundles ({view_mode})", leave=False):
        image = Image.open(image_map[bundle_id]).convert("RGB")
        views = generate_bundle_views(image, mode=view_mode)
        tensors = torch.stack([transform(view) for view in views], dim=0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            features = model(tensors)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        features = F.normalize(features, p=2, dim=1)
        out_ids.append(bundle_id)
        out_embeddings.append(features.cpu().numpy().astype(np.float32))

    return out_ids, out_embeddings


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


def build_section_priors(
    bundles_df: pd.DataFrame,
    products_df: pd.DataFrame,
    train_df: pd.DataFrame,
    encoded_product_ids: Sequence[str],
    top_descriptions_per_section: int,
) -> Tuple[Dict[str, str], Dict[str, np.ndarray], int]:
    """Build section->candidate product indices using description priors from train."""
    bundle_section_map = (
        bundles_df[["bundle_asset_id", "bundle_id_section"]]
        .dropna(subset=["bundle_asset_id"])
        .assign(bundle_asset_id=lambda df: df["bundle_asset_id"].astype(str))
        .assign(bundle_id_section=lambda df: df["bundle_id_section"].astype(str))
        .set_index("bundle_asset_id")["bundle_id_section"]
        .to_dict()
    )

    product_desc_map = (
        products_df[["product_asset_id", "product_description"]]
        .dropna(subset=["product_asset_id", "product_description"])
        .assign(product_asset_id=lambda df: df["product_asset_id"].astype(str))
        .assign(product_description=lambda df: df["product_description"].astype(str).str.strip().str.lower())
        .set_index("product_asset_id")["product_description"]
        .to_dict()
    )

    merged = train_df.copy()
    merged["bundle_asset_id"] = merged["bundle_asset_id"].astype(str)
    merged["product_asset_id"] = merged["product_asset_id"].astype(str)
    merged["section"] = merged["bundle_asset_id"].map(bundle_section_map)
    merged["description"] = merged["product_asset_id"].map(product_desc_map)
    merged = merged.dropna(subset=["section", "description"])

    section_desc_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in merged.itertuples(index=False):
        section_desc_counts[str(row.section)][str(row.description)] += 1

    section_top_desc: Dict[str, Set[str]] = {}
    for section, counter in section_desc_counts.items():
        top_desc = [desc for desc, _ in counter.most_common(max(1, top_descriptions_per_section))]
        section_top_desc[section] = set(top_desc)

    product_id_to_idx = {pid: idx for idx, pid in enumerate(encoded_product_ids)}
    section_to_indices: Dict[str, np.ndarray] = {}
    for section, desc_set in section_top_desc.items():
        indices = []
        for pid, desc in product_desc_map.items():
            if desc in desc_set and pid in product_id_to_idx:
                indices.append(product_id_to_idx[pid])
        if indices:
            section_to_indices[section] = np.array(sorted(set(indices)), dtype=np.int64)

    return bundle_section_map, section_to_indices, len(section_to_indices)


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Return sorted top-k indices from a 1D score vector."""
    if scores.ndim != 1:
        raise ValueError("scores must be 1D.")
    if k <= 0:
        raise ValueError("k must be > 0.")
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    order = np.argsort(-scores[idx])
    return idx[order]


def rank_products_for_bundle(
    bundle_view_embeddings: np.ndarray,
    product_embeddings: np.ndarray,
    product_ids: Sequence[str],
    top_k: int,
    score_agg: str,
    candidate_indices: np.ndarray | None = None,
) -> List[str]:
    """Rank products for one bundle using aggregated multi-view similarities."""
    if candidate_indices is not None and len(candidate_indices) > 0:
        candidate_emb = product_embeddings[candidate_indices]
        sims = bundle_view_embeddings @ candidate_emb.T
    else:
        candidate_indices = None
        sims = bundle_view_embeddings @ product_embeddings.T

    if score_agg == "mean":
        scores = sims.mean(axis=0)
    else:
        scores = sims.max(axis=0)

    top_local = topk_from_scores(scores, k=top_k)
    if candidate_indices is not None:
        top_global = candidate_indices[top_local]
    else:
        top_global = top_local
    return [str(product_ids[int(i)]) for i in top_global]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    eval_ks = parse_ks(args.eval_ks)
    top_n_submit = min(args.top_n_submit, 15)
    if top_n_submit <= 0:
        raise ValueError("--top-n-submit must be > 0")

    bundles_df = pd.read_csv(args.bundles_csv)
    products_df = pd.read_csv(args.products_csv)
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    bundle_image_map = build_image_map(args.bundle_images_dir)
    product_image_map = build_image_map(args.product_images_dir)

    product_ids_all = products_df["product_asset_id"].astype(str).tolist()
    product_ids = [pid for pid in product_ids_all if pid in product_image_map]
    if not product_ids:
        raise RuntimeError("No product images found for product ids.")

    device = resolve_device(args.device)
    model, transform, embedding_dim = build_pretrained_encoder(
        args.model_name,
        device=device,
        hf_model_id=args.hf_model_id,
    )
    model, device, used_gpus = maybe_wrap_dataparallel(model, device=device, gpu_ids=parse_gpu_ids(args.gpu_ids))

    if device.type == "cuda":
        if len(used_gpus) > 1:
            print(f"Using multi-GPU DataParallel on GPUs {used_gpus}")
        else:
            print(f"Using single GPU: {used_gpus[0] if used_gpus else 0}")
    else:
        print("Using CPU")

    print(f"Model={args.model_name} | products={len(product_ids)} | product_tta_flip={args.product_tta_flip}")
    encoded_product_ids, product_embeddings = encode_assets(
        asset_ids=product_ids,
        image_map=product_image_map,
        model=model,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embedding_dim=embedding_dim,
        use_amp=args.amp,
        tta_flip=args.product_tta_flip,
    )
    product_ids = encoded_product_ids
    print(f"Encoded product embeddings: {product_embeddings.shape}")

    bundle_section_map, section_to_candidate_indices, num_sections_with_prior = build_section_priors(
        bundles_df=bundles_df,
        products_df=products_df,
        train_df=train_df,
        encoded_product_ids=product_ids,
        top_descriptions_per_section=args.top_descriptions_per_section,
    )
    print(f"Section prior built for {num_sections_with_prior} sections")

    train_bundle_ids = train_df["bundle_asset_id"].astype(str).tolist()
    val_bundle_ids = split_val_bundles(train_bundle_ids, val_ratio=args.val_ratio, seed=args.seed)
    gt_map = build_gt_map(train_df)

    val_metrics: Dict[str, float] = {"num_bundles_evaluated": 0.0}
    if val_bundle_ids:
        print(f"Running validation on {len(val_bundle_ids)} bundles...")
        val_ids_encoded, val_views = encode_bundles_multiview(
            bundle_ids=val_bundle_ids,
            image_map=bundle_image_map,
            model=model,
            transform=transform,
            device=device,
            embedding_dim=embedding_dim,
            use_amp=args.amp,
            view_mode=args.bundle_view_mode,
        )
        val_predictions: List[List[str]] = []
        for bundle_id, view_embeddings in zip(val_ids_encoded, val_views):
            candidate_indices = None
            if args.use_section_prior:
                section = bundle_section_map.get(bundle_id)
                if section is not None:
                    candidate_indices = section_to_candidate_indices.get(str(section))
            pred_ids = rank_products_for_bundle(
                bundle_view_embeddings=view_embeddings,
                product_embeddings=product_embeddings,
                product_ids=product_ids,
                top_k=max(eval_ks),
                score_agg=args.score_agg,
                candidate_indices=candidate_indices,
            )
            val_predictions.append(pred_ids)

        val_metrics = evaluate_bundle_retrieval(
            bundle_ids=val_ids_encoded,
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

    print(
        f"Generating submission for {len(test_bundle_ids)} test bundles | "
        f"view_mode={args.bundle_view_mode} | score_agg={args.score_agg}"
    )
    test_ids_encoded, test_views = encode_bundles_multiview(
        bundle_ids=test_bundle_ids,
        image_map=bundle_image_map,
        model=model,
        transform=transform,
        device=device,
        embedding_dim=embedding_dim,
        use_amp=args.amp,
        view_mode=args.bundle_view_mode,
    )

    pred_map: Dict[str, List[str]] = {}
    for bundle_id, view_embeddings in zip(test_ids_encoded, test_views):
        candidate_indices = None
        if args.use_section_prior:
            section = bundle_section_map.get(bundle_id)
            if section is not None:
                candidate_indices = section_to_candidate_indices.get(str(section))
        pred_map[bundle_id] = rank_products_for_bundle(
            bundle_view_embeddings=view_embeddings,
            product_embeddings=product_embeddings,
            product_ids=product_ids,
            top_k=top_n_submit,
            score_agg=args.score_agg,
            candidate_indices=candidate_indices,
        )

    popular_products = train_df["product_asset_id"].astype(str).tolist()
    fallback_products = [pid for pid, _ in Counter(popular_products).most_common(top_n_submit)]
    if not fallback_products:
        fallback_products = product_ids[:top_n_submit]

    submission_rows: List[Dict[str, str]] = []
    for bundle_id in test_bundle_ids:
        preds = pred_map.get(bundle_id, fallback_products)[:top_n_submit]
        for product_id in preds:
            submission_rows.append({"bundle_asset_id": bundle_id, "product_asset_id": product_id})

    submission_df = pd.DataFrame(submission_rows, columns=["bundle_asset_id", "product_asset_id"])
    ensure_parent(args.submission_out)
    submission_df.to_csv(args.submission_out, index=False)

    missing_test_images = len(test_bundle_ids) - len(test_ids_encoded)
    summary = {
        "model_name": args.model_name,
        "hf_model_id": args.hf_model_id,
        "device": str(device),
        "used_gpus": used_gpus,
        "bundle_view_mode": args.bundle_view_mode,
        "score_agg": args.score_agg,
        "product_tta_flip": bool(args.product_tta_flip),
        "use_section_prior": bool(args.use_section_prior),
        "top_descriptions_per_section": int(args.top_descriptions_per_section),
        "num_products_indexed": int(len(product_ids)),
        "num_sections_with_prior": int(num_sections_with_prior),
        "num_test_bundles": int(len(test_bundle_ids)),
        "rows_written_submission": int(len(submission_df)),
        "missing_test_images": int(missing_test_images),
        "val_metrics": val_metrics,
    }
    ensure_parent(args.metrics_out)
    args.metrics_out.write_text(json.dumps(summary, indent=2))

    print(f"Saved submission: {args.submission_out} ({len(submission_df)} rows)")
    print(f"Saved metrics: {args.metrics_out}")
    print(f"Missing test bundle images handled with fallback: {missing_test_images}")


if __name__ == "__main__":
    main()
