#!/usr/bin/env python3
"""Check if product categories map consistently to bundle_id_section."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


SPLIT_RE = re.compile(r"[,;|/]+")
SPACE_RE = re.compile(r"\s+")
RARE_SEP_RE = re.compile(r"[_\t\r\n\-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether normalized product categories appear in a single "
            "bundle_id_section or across multiple sections."
        )
    )
    parser.add_argument(
        "--bundles-csv",
        type=Path,
        default=Path("data/bundles_dataset.csv"),
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/bundles_product_match_train.csv"),
    )
    parser.add_argument(
        "--products-csv",
        type=Path,
        default=Path("data/product_dataset.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/category_section_consistency.csv"),
        help="Detailed per-category report.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("outputs/category_section_summary.json"),
        help="Global summary stats for quick decision making.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=30,
        help="Minimum samples per category for reliability checks.",
    )
    parser.add_argument(
        "--purity-threshold",
        type=float,
        default=0.95,
        help="Threshold for almost-single-section categories.",
    )
    parser.add_argument(
        "--top-impure",
        type=int,
        default=20,
        help="How many impure categories to print in console.",
    )
    return parser.parse_args()


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    if not s:
        return ""
    s = RARE_SEP_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def extract_main_category(description: object) -> str:
    normalized = normalize_text(description)
    if not normalized:
        return ""
    category = SPLIT_RE.split(normalized)[0].strip()
    return category


def read_required_columns(path: Path, required: List[str], name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing columns {missing}. Found columns: {list(df.columns)}"
        )
    return df


def main() -> None:
    args = parse_args()

    if args.min_count <= 0:
        raise ValueError("--min-count must be > 0")
    if not 0.0 <= args.purity_threshold <= 1.0:
        raise ValueError("--purity-threshold must be in [0, 1]")
    if args.top_impure <= 0:
        raise ValueError("--top-impure must be > 0")

    bundles = read_required_columns(
        args.bundles_csv,
        ["bundle_asset_id", "bundle_id_section"],
        "bundles_csv",
    ).copy()
    train = read_required_columns(
        args.train_csv,
        ["bundle_asset_id", "product_asset_id"],
        "train_csv",
    ).copy()
    products = read_required_columns(
        args.products_csv,
        ["product_asset_id", "product_description"],
        "products_csv",
    ).copy()

    bundles["bundle_asset_id"] = bundles["bundle_asset_id"].astype(str)
    train["bundle_asset_id"] = train["bundle_asset_id"].astype(str)
    train["product_asset_id"] = train["product_asset_id"].astype(str)
    products["product_asset_id"] = products["product_asset_id"].astype(str)

    # Build product -> normalized category mapping.
    products["product_category"] = products["product_description"].map(extract_main_category)
    products = products[products["product_category"] != ""].copy()

    merged = (
        train.merge(
            bundles[["bundle_asset_id", "bundle_id_section"]],
            on="bundle_asset_id",
            how="inner",
        )
        .merge(
            products[["product_asset_id", "product_category"]],
            on="product_asset_id",
            how="inner",
        )
    )
    if merged.empty:
        raise RuntimeError("No merged rows found after joining train + bundles + products.")

    merged["bundle_id_section"] = merged["bundle_id_section"].astype(str)

    counts = (
        merged.groupby(["product_category", "bundle_id_section"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    total = (
        counts.groupby("product_category", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "total_count"})
    )
    unique_sections = (
        counts.groupby("product_category", as_index=False)["bundle_id_section"]
        .nunique()
        .rename(columns={"bundle_id_section": "num_sections"})
    )

    dominant = (
        counts.sort_values(["product_category", "count"], ascending=[True, False])
        .drop_duplicates(subset=["product_category"], keep="first")
        .rename(
            columns={
                "bundle_id_section": "dominant_section",
                "count": "dominant_count",
            }
        )[["product_category", "dominant_section", "dominant_count"]]
    )

    section_distribution: Dict[str, Dict[str, int]] = {}
    for row in counts.itertuples(index=False):
        section_distribution.setdefault(row.product_category, {})[row.bundle_id_section] = int(row.count)

    report = (
        total.merge(unique_sections, on="product_category", how="left")
        .merge(dominant, on="product_category", how="left")
        .sort_values(["total_count", "dominant_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    report["dominant_ratio"] = report["dominant_count"] / report["total_count"]
    report["is_single_section"] = report["num_sections"] == 1
    report["is_almost_single_section"] = report["dominant_ratio"] >= args.purity_threshold
    report["sections_count_json"] = report["product_category"].map(
        lambda c: json.dumps(section_distribution.get(c, {}), sort_keys=True)
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out_csv, index=False)

    reliable = report[report["total_count"] >= args.min_count]
    summary = {
        "num_rows_joined": int(len(merged)),
        "num_categories_total": int(len(report)),
        "num_categories_min_count": int(len(reliable)),
        "min_count": int(args.min_count),
        "purity_threshold": float(args.purity_threshold),
        "single_section_categories_total": int(report["is_single_section"].sum()),
        "single_section_categories_min_count": int(reliable["is_single_section"].sum()),
        "almost_single_categories_min_count": int(reliable["is_almost_single_section"].sum()),
        "single_section_rate_min_count": (
            float(reliable["is_single_section"].mean()) if len(reliable) else 0.0
        ),
        "almost_single_rate_min_count": (
            float(reliable["is_almost_single_section"].mean()) if len(reliable) else 0.0
        ),
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    impure = reliable[~reliable["is_single_section"]].head(args.top_impure)
    print("=== Category vs Section Consistency ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved detailed report: {args.out_csv}")
    print(f"Saved summary: {args.out_summary}")
    if not impure.empty:
        print("\nTop impure categories (min_count filtered):")
        cols = [
            "product_category",
            "total_count",
            "num_sections",
            "dominant_section",
            "dominant_ratio",
            "sections_count_json",
        ]
        print(impure[cols].to_string(index=False))


if __name__ == "__main__":
    main()
