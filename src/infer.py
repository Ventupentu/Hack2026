"""Baseline pretrained retrieval for bundle -> products prediction."""

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
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise ModuleNotFoundError(
        "Missing PyTorch dependencies. Install with:\n"
        "  pip install torch torchvision"
    ) from exc

from src.models.feature_extractors import build_pretrained_encoder, resolve_device
from src.utils.metrics import evaluate_bundle_retrieval
from src.utils.retrieval import retrieve_topk_product_ids


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
    parser = argparse.ArgumentParser(description="Pretrained visual retrieval baseline")
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
        default="resnet50",
        choices=["resnet18", "resnet50", "efficientnet_b0"],
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ks", type=str, default="1,5,10,15")
    parser.add_argument("--top-n-submit", type=int, default=15)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
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


@torch.inference_mode()
def encode_assets(
    asset_ids: Sequence[str],
    image_map: Dict[str, Path],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    embedding_dim: int,
    use_amp: bool,
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
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        features = F.normalize(features, p=2, dim=1)
        output_embeddings.append(features.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(output_embeddings, axis=0)
    return output_ids, embeddings


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
    model, transform, embedding_dim = build_pretrained_encoder(args.model_name, device=device)

    print(f"Using model={args.model_name} | device={device} | products={len(product_ids)}")
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
    )
    product_ids = encoded_product_ids
    print(f"Encoded product embeddings: {product_embeddings.shape}")

    train_bundle_ids = train_df["bundle_asset_id"].astype(str).tolist()
    val_bundle_ids = split_val_bundles(train_bundle_ids, val_ratio=args.val_ratio, seed=args.seed)
    gt_map = build_gt_map(train_df)

    val_metrics: Dict[str, float] = {"num_bundles_evaluated": 0.0}
    if val_bundle_ids:
        print(f"Running validation on {len(val_bundle_ids)} bundles...")
        val_ids_encoded, val_embeddings = encode_assets(
            asset_ids=val_bundle_ids,
            image_map=bundle_image_map,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            embedding_dim=embedding_dim,
            use_amp=args.amp,
        )
        val_predictions = retrieve_topk_product_ids(
            query_embeddings=val_embeddings,
            product_embeddings=product_embeddings,
            product_ids=product_ids,
            k=max(eval_ks),
        )
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
    print(f"Generating submission for {len(test_bundle_ids)} test bundles...")
    test_ids_encoded, test_embeddings = encode_assets(
        asset_ids=test_bundle_ids,
        image_map=bundle_image_map,
        model=model,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embedding_dim=embedding_dim,
        use_amp=args.amp,
    )
    test_predictions = retrieve_topk_product_ids(
        query_embeddings=test_embeddings,
        product_embeddings=product_embeddings,
        product_ids=product_ids,
        k=top_n_submit,
    )
    pred_map = {bundle_id: preds for bundle_id, preds in zip(test_ids_encoded, test_predictions)}

    popular_products = train_df["product_asset_id"].astype(str).tolist()
    fallback_products = [pid for pid, _ in Counter(popular_products).most_common(top_n_submit)]
    if not fallback_products:
        fallback_products = product_ids[:top_n_submit]

    submission_rows: List[Dict[str, str]] = []
    for bundle_id in test_bundle_ids:
        preds = pred_map.get(bundle_id, fallback_products)[:top_n_submit]
        for product_id in preds:
            submission_rows.append(
                {"bundle_asset_id": bundle_id, "product_asset_id": product_id}
            )

    submission_df = pd.DataFrame(submission_rows, columns=["bundle_asset_id", "product_asset_id"])
    ensure_parent(args.submission_out)
    submission_df.to_csv(args.submission_out, index=False)

    summary = {
        "model_name": args.model_name,
        "device": str(device),
        "num_products_indexed": int(len(product_ids)),
        "num_test_bundles": int(len(test_bundle_ids)),
        "rows_written_submission": int(len(submission_df)),
        "val_metrics": val_metrics,
    }
    ensure_parent(args.metrics_out)
    args.metrics_out.write_text(json.dumps(summary, indent=2))

    missing_test_images = len(test_bundle_ids) - len(test_ids_encoded)
    print(f"Saved submission: {args.submission_out} ({len(submission_df)} rows)")
    print(f"Saved metrics: {args.metrics_out}")
    print(f"Missing test bundle images handled with fallback: {missing_test_images}")


if __name__ == "__main__":
    main()
