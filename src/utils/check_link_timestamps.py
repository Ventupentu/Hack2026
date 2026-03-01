#!/usr/bin/env python3
"""Compare bundle/product URL ts periods aggregated per bundle.

Reads:
- bundles_dataset.csv (bundle_asset_id, bundle_image_url)
- product_dataset.csv (product_asset_id, product_image_url)
- bundles_product_match_train.csv (bundle_asset_id, product_asset_id)

Outputs a CSV with one row per bundle, evaluating all its matched products
and whether their link timestamps coincide with bundle date/month/quarter.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


def _extract_ts(url: object) -> Optional[int]:
    """Extract URL query parameter ts as integer timestamp."""
    if pd.isna(url):
        return None
    text = str(url).strip()
    if not text:
        return None
    query = parse_qs(urlparse(text).query)
    raw = query.get("ts")
    if not raw:
        return None
    value = str(raw[0]).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalize_to_seconds(ts_value: int) -> float:
    """Normalize ts to Unix seconds.

    Zara links use milliseconds, but we keep this robust:
    - >= 1e12: milliseconds
    - else: seconds
    """
    return float(ts_value) / 1000.0 if ts_value >= 10**12 else float(ts_value)


def _to_datetime_fields(ts_value: Optional[int], timezone_name: str) -> Tuple[str, str, str, str]:
    """Return datetime/date/month/quarter fields for ts."""
    if ts_value is None:
        return "", "", "", ""
    try:
        if ts_value >= 10**12:
            ts = pd.Timestamp(ts_value, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(ts_value, unit="s", tz="UTC")
    except (ValueError, OverflowError):
        return "", ""

    if timezone_name.upper() != "UTC":
        if ZoneInfo is None:
            raise RuntimeError("Timezone conversion needs zoneinfo support in this Python version.")
        ts = ts.tz_convert(ZoneInfo(timezone_name))

    return (
        ts.isoformat(),
        ts.date().isoformat(),
        f"{ts.year:04d}-{ts.month:02d}",
        f"{ts.year:04d}-Q{((ts.month - 1) // 3) + 1}",
    )


def _safe_delta_hours(bundle_ts: Optional[int], product_ts: Optional[int]) -> float:
    if bundle_ts is None or product_ts is None:
        return math.nan
    b_sec = _normalize_to_seconds(bundle_ts)
    p_sec = _normalize_to_seconds(product_ts)
    return abs(b_sec - p_sec) / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check ts date/month/quarter alignment between bundles and all matched products."
    )
    parser.add_argument("--bundles-csv", type=Path, default=Path("data/bundles_dataset.csv"))
    parser.add_argument("--products-csv", type=Path, default=Path("data/product_dataset.csv"))
    parser.add_argument("--matches-csv", type=Path, default=Path("data/bundles_product_match_train.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/ts_date_alignment_report.csv"))
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="Timezone for calendar-date comparison (e.g. UTC, Europe/Madrid).",
    )
    args = parser.parse_args()

    bundles = pd.read_csv(args.bundles_csv, usecols=["bundle_asset_id", "bundle_image_url"])
    products = pd.read_csv(args.products_csv, usecols=["product_asset_id", "product_image_url"])
    matches = pd.read_csv(args.matches_csv, usecols=["bundle_asset_id", "product_asset_id"])

    bundles["bundle_asset_id"] = bundles["bundle_asset_id"].astype(str)
    products["product_asset_id"] = products["product_asset_id"].astype(str)
    matches["bundle_asset_id"] = matches["bundle_asset_id"].astype(str)
    matches["product_asset_id"] = matches["product_asset_id"].astype(str)

    bundles["bundle_ts"] = bundles["bundle_image_url"].map(_extract_ts)
    products["product_ts"] = products["product_image_url"].map(_extract_ts)

    matches = matches.drop_duplicates(subset=["bundle_asset_id", "product_asset_id"], keep="first")

    merged = matches.merge(
        bundles[["bundle_asset_id", "bundle_ts"]], on="bundle_asset_id", how="left"
    )
    merged = merged.merge(products[["product_asset_id", "product_ts"]], on="product_asset_id", how="left")

    bundle_dt = merged["bundle_ts"].map(lambda x: _to_datetime_fields(x, args.timezone))
    product_dt = merged["product_ts"].map(lambda x: _to_datetime_fields(x, args.timezone))
    merged["bundle_datetime"] = [item[0] for item in bundle_dt]
    merged["bundle_date"] = [item[1] for item in bundle_dt]
    merged["bundle_month"] = [item[2] for item in bundle_dt]
    merged["bundle_quarter"] = [item[3] for item in bundle_dt]
    merged["product_datetime"] = [item[0] for item in product_dt]
    merged["product_date"] = [item[1] for item in product_dt]
    merged["product_month"] = [item[2] for item in product_dt]
    merged["product_quarter"] = [item[3] for item in product_dt]

    merged["same_date"] = (
        (merged["bundle_date"] != "")
        & (merged["product_date"] != "")
        & (merged["bundle_date"] == merged["product_date"])
    )
    merged["same_month"] = (
        (merged["bundle_month"] != "")
        & (merged["product_month"] != "")
        & (merged["bundle_month"] == merged["product_month"])
    )
    merged["same_quarter"] = (
        (merged["bundle_quarter"] != "")
        & (merged["product_quarter"] != "")
        & (merged["bundle_quarter"] == merged["product_quarter"])
    )
    merged["delta_hours_abs"] = [
        _safe_delta_hours(b_ts, p_ts)
        for b_ts, p_ts in zip(merged["bundle_ts"], merged["product_ts"])
    ]

    grouped_rows = []
    for bundle_id, group in merged.groupby("bundle_asset_id", sort=True):
        n_products = int(len(group))
        both_present_mask = group["bundle_ts"].notna() & group["product_ts"].notna()
        n_products_with_ts = int(both_present_mask.sum())
        n_products_same_date = int(group.loc[both_present_mask, "same_date"].sum())
        n_products_diff_date = n_products_with_ts - n_products_same_date
        n_products_same_month = int(group.loc[both_present_mask, "same_month"].sum())
        n_products_diff_month = n_products_with_ts - n_products_same_month
        n_products_same_quarter = int(group.loc[both_present_mask, "same_quarter"].sum())
        n_products_diff_quarter = n_products_with_ts - n_products_same_quarter
        n_products_missing_ts = n_products - n_products_with_ts

        bundle_ts = group["bundle_ts"].iloc[0]
        bundle_datetime = group["bundle_datetime"].iloc[0]
        bundle_date = group["bundle_date"].iloc[0]
        bundle_month = group["bundle_month"].iloc[0]
        bundle_quarter = group["bundle_quarter"].iloc[0]

        product_ids_same_date = group.loc[group["same_date"], "product_asset_id"].astype(str).tolist()
        product_ids_diff_date = group.loc[both_present_mask & ~group["same_date"], "product_asset_id"].astype(str).tolist()
        product_ids_same_month = group.loc[group["same_month"], "product_asset_id"].astype(str).tolist()
        product_ids_diff_month = group.loc[both_present_mask & ~group["same_month"], "product_asset_id"].astype(str).tolist()
        product_ids_same_quarter = group.loc[group["same_quarter"], "product_asset_id"].astype(str).tolist()
        product_ids_diff_quarter = group.loc[both_present_mask & ~group["same_quarter"], "product_asset_id"].astype(str).tolist()
        product_ids_missing_ts = group.loc[~both_present_mask, "product_asset_id"].astype(str).tolist()

        avg_delta_hours = group.loc[both_present_mask, "delta_hours_abs"].mean()
        max_delta_hours = group.loc[both_present_mask, "delta_hours_abs"].max()

        grouped_rows.append(
            {
                "bundle_asset_id": bundle_id,
                "bundle_ts": bundle_ts,
                "bundle_datetime": bundle_datetime,
                "bundle_date": bundle_date,
                "bundle_month": bundle_month,
                "bundle_quarter": bundle_quarter,
                "n_products": n_products,
                "n_products_with_ts": n_products_with_ts,
                "n_products_missing_ts": n_products_missing_ts,
                "n_products_same_date": n_products_same_date,
                "n_products_diff_date": n_products_diff_date,
                "pct_products_same_date": (
                    (n_products_same_date / n_products_with_ts) if n_products_with_ts > 0 else math.nan
                ),
                "all_products_same_date": (
                    n_products_with_ts > 0 and n_products_same_date == n_products_with_ts
                ),
                "any_product_same_date": n_products_same_date > 0,
                "n_products_same_month": n_products_same_month,
                "n_products_diff_month": n_products_diff_month,
                "pct_products_same_month": (
                    (n_products_same_month / n_products_with_ts) if n_products_with_ts > 0 else math.nan
                ),
                "all_products_same_month": (
                    n_products_with_ts > 0 and n_products_same_month == n_products_with_ts
                ),
                "any_product_same_month": n_products_same_month > 0,
                "n_products_same_quarter": n_products_same_quarter,
                "n_products_diff_quarter": n_products_diff_quarter,
                "pct_products_same_quarter": (
                    (n_products_same_quarter / n_products_with_ts) if n_products_with_ts > 0 else math.nan
                ),
                "all_products_same_quarter": (
                    n_products_with_ts > 0 and n_products_same_quarter == n_products_with_ts
                ),
                "any_product_same_quarter": n_products_same_quarter > 0,
                "avg_delta_hours_abs": avg_delta_hours,
                "max_delta_hours_abs": max_delta_hours,
                "product_ids_same_date": "|".join(product_ids_same_date),
                "product_ids_diff_date": "|".join(product_ids_diff_date),
                "product_ids_same_month": "|".join(product_ids_same_month),
                "product_ids_diff_month": "|".join(product_ids_diff_month),
                "product_ids_same_quarter": "|".join(product_ids_same_quarter),
                "product_ids_diff_quarter": "|".join(product_ids_diff_quarter),
                "product_ids_missing_ts": "|".join(product_ids_missing_ts),
            }
        )

    out = pd.DataFrame(grouped_rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    total_bundles = len(out)
    bundles_all_same = int(out["all_products_same_date"].sum())
    bundles_any_same = int(out["any_product_same_date"].sum())
    bundles_none_same = total_bundles - bundles_any_same
    total_products = int(out["n_products"].sum())
    total_products_with_ts = int(out["n_products_with_ts"].sum())
    total_products_same = int(out["n_products_same_date"].sum())
    total_products_same_month = int(out["n_products_same_month"].sum())
    total_products_same_quarter = int(out["n_products_same_quarter"].sum())
    bundles_all_same_month = int(out["all_products_same_month"].sum())
    bundles_any_same_month = int(out["any_product_same_month"].sum())
    bundles_all_same_quarter = int(out["all_products_same_quarter"].sum())
    bundles_any_same_quarter = int(out["any_product_same_quarter"].sum())

    print(f"Bundles analyzed: {total_bundles}")
    print(f"Products linked (total): {total_products}")
    print(f"Products with ts present: {total_products_with_ts}")
    print(f"Products same date ({args.timezone}): {total_products_same}")
    print(f"Products same month ({args.timezone}): {total_products_same_month}")
    print(f"Products same quarter ({args.timezone}): {total_products_same_quarter}")
    print(f"Bundles where ALL products match date: {bundles_all_same}")
    print(f"Bundles where AT LEAST ONE product matches date: {bundles_any_same}")
    print(f"Bundles where NO products match date: {bundles_none_same}")
    print(f"Bundles where ALL products match month: {bundles_all_same_month}")
    print(f"Bundles where AT LEAST ONE product matches month: {bundles_any_same_month}")
    print(f"Bundles where ALL products match quarter: {bundles_all_same_quarter}")
    print(f"Bundles where AT LEAST ONE product matches quarter: {bundles_any_same_quarter}")
    print(f"Saved report: {args.out_csv}")


if __name__ == "__main__":
    main()
