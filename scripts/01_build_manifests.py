#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, setup_logging, write_jsonl

LOGGER = logging.getLogger("build_manifests")


def resolve_default_path(preferred: str, alternatives: list[str]) -> Path:
    p = ROOT / preferred
    if p.exists():
        return p
    for alt in alternatives:
        cand = ROOT / alt
        if cand.exists():
            return cand
    return p


def build_strata(counts: pd.Series) -> pd.Series:
    edges = [0, 1, 2, 3, 4, 5, 8, 12, 100]
    bins = pd.cut(counts, bins=edges, labels=False, include_lowest=True)
    bins = bins.fillna(0).astype(int)

    value_counts = bins.value_counts()
    if (value_counts < 2).any():
        # Fallback to quantile bins if fixed bins are too sparse.
        try:
            bins = pd.qcut(counts.rank(method="first"), q=min(5, len(counts)), labels=False, duplicates="drop")
            bins = bins.fillna(0).astype(int)
        except ValueError:
            bins = pd.Series(np.zeros(len(counts), dtype=int), index=counts.index)
    return bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build products/train/val manifests")
    parser.add_argument(
        "--bundles_csv",
        type=Path,
        default=resolve_default_path("data/bundles.csv", ["data/bundles_dataset.csv"]),
    )
    parser.add_argument(
        "--products_csv",
        type=Path,
        default=resolve_default_path("data/products.csv", ["data/product_dataset.csv"]),
    )
    parser.add_argument(
        "--train_relations_csv",
        type=Path,
        default=resolve_default_path("data/train_relations.csv", ["data/bundles_product_match_train.csv"]),
    )
    parser.add_argument("--bundle_paths_csv", type=Path, default=Path("artifacts/paths/bundle_paths.csv"))
    parser.add_argument("--product_paths_csv", type=Path, default=Path("artifacts/paths/product_paths.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dir(args.output_dir)

    bundles = pd.read_csv(args.bundles_csv)
    products = pd.read_csv(args.products_csv)
    relations = pd.read_csv(args.train_relations_csv)

    if args.bundle_paths_csv.exists():
        bundle_paths = pd.read_csv(args.bundle_paths_csv)
        if "bundle_asset_id" not in bundle_paths.columns:
            raise ValueError("bundle_paths_csv must contain bundle_asset_id")
    else:
        bundle_paths = pd.DataFrame({
            "bundle_asset_id": bundles["bundle_asset_id"].astype(str),
            "image_path": bundles["bundle_asset_id"].astype(str).map(lambda x: str(Path("data/bundle_images") / f"{x}.jpg")),
            "success": 1,
        })

    if args.product_paths_csv.exists():
        product_paths = pd.read_csv(args.product_paths_csv)
        if "product_asset_id" not in product_paths.columns:
            raise ValueError("product_paths_csv must contain product_asset_id")
    else:
        product_paths = pd.DataFrame({
            "product_asset_id": products["product_asset_id"].astype(str),
            "image_path": products["product_asset_id"].astype(str).map(lambda x: str(Path("data/product_images") / f"{x}.jpg")),
            "success": 1,
        })

    # Keep only assets with valid local paths.
    bundle_paths = bundle_paths[bundle_paths.get("success", 1) == 1].copy()
    product_paths = product_paths[product_paths.get("success", 1) == 1].copy()
    bundle_paths = bundle_paths[bundle_paths["image_path"].map(lambda p: Path(str(p)).exists())]
    product_paths = product_paths[product_paths["image_path"].map(lambda p: Path(str(p)).exists())]

    bundle_paths["bundle_asset_id"] = bundle_paths["bundle_asset_id"].astype(str)
    product_paths["product_asset_id"] = product_paths["product_asset_id"].astype(str)

    bundles["bundle_asset_id"] = bundles["bundle_asset_id"].astype(str)
    products["product_asset_id"] = products["product_asset_id"].astype(str)
    relations["bundle_asset_id"] = relations["bundle_asset_id"].astype(str)
    relations["product_asset_id"] = relations["product_asset_id"].astype(str)

    # Products manifest.
    valid_product_ids = set(product_paths["product_asset_id"].tolist())
    products_manifest_rows = []
    for row in product_paths.itertuples(index=False):
        pid = str(row.product_asset_id)
        if pid not in valid_product_ids:
            continue
        products_manifest_rows.append(
            {
                "product_asset_id": pid,
                "image_path": str(row.image_path),
            }
        )

    products_manifest_path = args.output_dir / "products_manifest.jsonl"
    write_jsonl(products_manifest_path, products_manifest_rows)

    # Bundle manifests with grouped positives.
    valid_bundle_ids = set(bundle_paths["bundle_asset_id"].tolist())
    valid_product_ids = set([row["product_asset_id"] for row in products_manifest_rows])

    relations = relations[
        relations["bundle_asset_id"].isin(valid_bundle_ids)
        & relations["product_asset_id"].isin(valid_product_ids)
    ].copy()

    grouped = relations.groupby("bundle_asset_id")["product_asset_id"].apply(lambda s: sorted(set(s.tolist())))

    bundle_path_map = dict(zip(bundle_paths["bundle_asset_id"], bundle_paths["image_path"]))
    manifest_rows = []
    for bundle_id, positives in grouped.items():
        image_path = bundle_path_map.get(bundle_id)
        if not image_path:
            continue
        manifest_rows.append(
            {
                "bundle_asset_id": bundle_id,
                "image_path": str(image_path),
                "positives": positives,
                "n_positives": len(positives),
            }
        )

    if not manifest_rows:
        raise RuntimeError("No manifest rows generated. Check CSV paths and downloaded assets.")

    manifest_df = pd.DataFrame(manifest_rows)
    strata = build_strata(manifest_df["n_positives"])

    try:
        train_df, val_df = train_test_split(
            manifest_df,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=strata,
        )
    except ValueError:
        LOGGER.warning("Stratified split failed; falling back to random split")
        train_df, val_df = train_test_split(
            manifest_df,
            test_size=args.val_ratio,
            random_state=args.seed,
            shuffle=True,
        )

    train_rows = train_df.to_dict(orient="records")
    val_rows = val_df.to_dict(orient="records")

    train_manifest_path = args.output_dir / "train_manifest.jsonl"
    val_manifest_path = args.output_dir / "val_manifest.jsonl"

    write_jsonl(train_manifest_path, train_rows)
    write_jsonl(val_manifest_path, val_rows)

    split_stats = {
        "num_products": len(products_manifest_rows),
        "num_train_bundles": len(train_rows),
        "num_val_bundles": len(val_rows),
        "avg_train_positives": float(train_df["n_positives"].mean()),
        "avg_val_positives": float(val_df["n_positives"].mean()),
        "seed": args.seed,
    }
    (args.output_dir / "split_stats.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")

    LOGGER.info("Saved products manifest: %s", products_manifest_path)
    LOGGER.info("Saved train manifest: %s", train_manifest_path)
    LOGGER.info("Saved val manifest: %s", val_manifest_path)
    LOGGER.info("Split stats: %s", json.dumps(split_stats))


if __name__ == "__main__":
    main()
