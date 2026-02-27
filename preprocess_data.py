#!/usr/bin/env python3
"""Preprocess bundle/product CSVs into multi-label manifests."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm


LOGGER = logging.getLogger("preprocess_data")
SPLIT_RE = re.compile(r"[,;|/]+")
SPACE_RE = re.compile(r"\s+")
RARE_SEP_RE = re.compile(r"[_\t\r\n\-]+")


@dataclass(frozen=True)
class BundleRecord:
    """Single bundle sample for multi-label training."""

    bundle_asset_id: str
    section_id: str
    image_url: str
    labels: List[str]


def configure_logging() -> None:
    """Configure standard logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_csv_with_schema(path: Path, required_columns: Sequence[str], name: str) -> pd.DataFrame:
    """Read CSV and validate required columns."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{name} not found: {path}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed reading {name} ({path}): {exc}") from exc

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def clean_and_split_labels(description: object) -> List[str]:
    """Normalize product_description into cleaned label tokens."""
    if pd.isna(description):
        return []
    text = str(description).strip().lower()
    if not text:
        return []
    text = RARE_SEP_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    parts = SPLIT_RE.split(text)
    labels: List[str] = []
    for part in parts:
        token = SPACE_RE.sub(" ", part).strip()
        if token:
            labels.append(token)
    return labels


def dedupe_keep_first(values: Iterable[str]) -> List[str]:
    """Deduplicate preserving first appearance."""
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def ensure_dir(path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def build_bundle_records(
    bundles_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    products_df: pd.DataFrame,
    strict: bool,
    limit: Optional[int],
) -> Tuple[List[BundleRecord], Dict[str, int]]:
    """Build training records and integrity counters."""
    counts: Dict[str, int] = defaultdict(int)

    bundles_df = bundles_df.copy()
    products_df = products_df.copy()
    relations_df = relations_df.copy()

    for df, key, name in (
        (bundles_df, "bundle_asset_id", "bundles"),
        (products_df, "product_asset_id", "products"),
    ):
        dup_count = int(df.duplicated(subset=[key]).sum())
        if dup_count:
            LOGGER.warning("%s: found %d duplicated %s, keeping first", name, dup_count, key)
            counts[f"duplicates_{key}"] += dup_count
            df.drop_duplicates(subset=[key], keep="first", inplace=True)

    bundle_map = bundles_df.set_index("bundle_asset_id")[
        [col for col in ("bundle_id_section", "bundle_image_url") if col in bundles_df.columns]
    ].to_dict(orient="index")
    product_desc_map = products_df.set_index("product_asset_id")["product_description"].to_dict()

    relations_df["bundle_asset_id"] = relations_df["bundle_asset_id"].astype(str)
    relations_df["product_asset_id"] = relations_df["product_asset_id"].astype(str)

    valid_bundle_ids = set(bundle_map.keys())
    valid_product_ids = set(product_desc_map.keys())

    missing_bundle_mask = ~relations_df["bundle_asset_id"].isin(valid_bundle_ids)
    missing_product_mask = ~relations_df["product_asset_id"].isin(valid_product_ids)

    missing_bundles = int(missing_bundle_mask.sum())
    missing_products = int(missing_product_mask.sum())
    if missing_bundles:
        LOGGER.warning(
            "Relations with unknown bundle_asset_id: %d (will discard%s)",
            missing_bundles,
            " and fail in --strict mode" if strict else "",
        )
    if missing_products:
        LOGGER.warning(
            "Relations with unknown product_asset_id: %d (will discard%s)",
            missing_products,
            " and fail in --strict mode" if strict else "",
        )
    counts["relations_missing_bundle"] = missing_bundles
    counts["relations_missing_product"] = missing_products

    if strict and (missing_bundles or missing_products):
        raise ValueError(
            "Integrity errors in train_relations: "
            f"missing bundles={missing_bundles}, missing products={missing_products}"
        )

    valid_relations = relations_df.loc[~missing_bundle_mask & ~missing_product_mask].copy()
    counts["relations_total"] = int(len(relations_df))
    counts["relations_valid"] = int(len(valid_relations))

    labels_by_bundle: Dict[str, List[str]] = defaultdict(list)
    missing_description_count = 0
    empty_after_clean_count = 0

    for row in valid_relations.itertuples(index=False):
        bundle_id = row.bundle_asset_id
        product_id = row.product_asset_id
        description = product_desc_map.get(product_id)
        if pd.isna(description):
            missing_description_count += 1
            continue
        labels = clean_and_split_labels(description)
        if not labels:
            empty_after_clean_count += 1
            continue
        labels_by_bundle[bundle_id].extend(labels)

    counts["products_missing_description"] = missing_description_count
    counts["products_empty_labels_after_clean"] = empty_after_clean_count

    records: List[BundleRecord] = []
    bundles_without_labels = 0
    for bundle_id, labels in labels_by_bundle.items():
        clean_labels = sorted(set(dedupe_keep_first(labels)))
        if not clean_labels:
            bundles_without_labels += 1
            continue
        bundle_meta = bundle_map[bundle_id]
        section = bundle_meta.get("bundle_id_section")
        image_url = bundle_meta.get("bundle_image_url")
        section_str = "" if pd.isna(section) else str(section)
        image_url_str = "" if pd.isna(image_url) else str(image_url).strip()
        records.append(
            BundleRecord(
                bundle_asset_id=str(bundle_id),
                section_id=section_str,
                image_url=image_url_str,
                labels=clean_labels,
            )
        )

    counts["bundles_without_labels"] = bundles_without_labels
    counts["bundles_with_labels"] = len(records)
    records.sort(key=lambda x: x.bundle_asset_id)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        records = records[:limit]
        counts["bundles_after_limit"] = len(records)

    LOGGER.info(
        "Built %d bundle records with labels from %d valid relations",
        len(records),
        counts["relations_valid"],
    )
    return records, counts


def build_label_vocab(records: Sequence[BundleRecord]) -> Tuple[List[str], Dict[str, int]]:
    """Create deterministic vocabulary and mapping."""
    vocab = sorted({label for record in records for label in record.labels})
    label2idx = {label: idx for idx, label in enumerate(vocab)}
    return vocab, label2idx


def split_train_val(
    records: Sequence[BundleRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[BundleRecord], List[BundleRecord]]:
    """Approximate stratified split by number of labels per bundle."""
    if not 0 < val_ratio < 1:
        raise ValueError("--val_ratio must be between 0 and 1")

    grouped: Dict[int, List[BundleRecord]] = defaultdict(list)
    for record in records:
        grouped[len(record.labels)].append(record)

    rng = random.Random(seed)
    train_records: List[BundleRecord] = []
    val_records: List[BundleRecord] = []

    for _, group in sorted(grouped.items(), key=lambda x: x[0]):
        items = list(group)
        items.sort(key=lambda x: x.bundle_asset_id)
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        n_val = min(max(n_val, 0), len(items))
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    train_records.sort(key=lambda x: x.bundle_asset_id)
    val_records.sort(key=lambda x: x.bundle_asset_id)
    return train_records, val_records


def download_image(
    url: str,
    out_path: Path,
    timeout: int,
    retries: int,
) -> Tuple[bool, Optional[str]]:
    """Download one image with retries.

    Returns:
        (download_ok, forbidden_url_if_403)
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, None
    if not url:
        return False, None

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            ensure_dir(out_path.parent)
            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            tmp_path.replace(out_path)
            return True, None
        except requests.exceptions.HTTPError as exc:  # pragma: no cover - network dependent
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            status_code = exc.response.status_code if exc.response is not None else None
            if attempt < retries:
                continue
            LOGGER.warning("HTTP error downloading %s -> %s (status=%s)", url, out_path, status_code)
            if status_code == 403:
                return False, url
            return False, None
        except Exception as exc:  # pragma: no cover - network dependent
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < retries:
                continue
            LOGGER.warning("Failed to download %s -> %s (%s)", url, out_path, exc)
            return False, None
    return False, None


def download_many_images(
    tasks: Sequence[Tuple[str, Path, str]],
    max_workers: int,
    timeout: int,
    retries: int,
) -> Tuple[Dict[str, bool], List[str]]:
    """Download images concurrently.

    Returns:
        (status_by_item_id, forbidden_urls)
    """
    results: Dict[str, bool] = {}
    forbidden_urls: List[str] = []
    if not tasks:
        return results, forbidden_urls

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(download_image, url, path, timeout, retries): (item_id, path)
            for item_id, path, url in tasks
        }
        for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Downloading"):
            item_id, _ = future_to_item[future]
            ok = False
            try:
                ok, forbidden_url = future.result()
                if forbidden_url:
                    forbidden_urls.append(forbidden_url)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("Download worker failed for id=%s: %s", item_id, exc)
            results[item_id] = ok
    return results, forbidden_urls


def write_manifest(
    path: Path,
    records: Sequence[BundleRecord],
    label2idx: Dict[str, int],
    image_status: Dict[str, bool],
    bundle_img_dir: Path,
) -> None:
    """Write JSONL manifest."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            image_path = bundle_img_dir / f"{record.bundle_asset_id}.jpg"
            payload = {
                "bundle_asset_id": record.bundle_asset_id,
                "section_id": record.section_id,
                "image_path": str(image_path.resolve()),
                "labels": record.labels,
                "label_indices": [label2idx[label] for label in record.labels],
            }
            if not image_status.get(record.bundle_asset_id, image_path.exists()):
                LOGGER.debug("Bundle image missing for %s", record.bundle_asset_id)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def compute_stats(
    records: Sequence[BundleRecord],
    train_records: Sequence[BundleRecord],
    val_records: Sequence[BundleRecord],
    label_vocab: Sequence[str],
    image_status: Dict[str, bool],
) -> Dict[str, object]:
    """Compute preprocessing stats."""
    label_frequency: Counter[str] = Counter()
    label_count_dist: Counter[int] = Counter()
    for record in records:
        label_frequency.update(record.labels)
        label_count_dist[len(record.labels)] += 1

    total = len(records)
    downloaded_ok = sum(1 for r in records if image_status.get(r.bundle_asset_id, False))
    pct_download_ok = (downloaded_ok / total * 100.0) if total else 0.0

    return {
        "num_bundles_total": total,
        "num_bundles_train": len(train_records),
        "num_bundles_val": len(val_records),
        "num_labels": len(label_vocab),
        "label_frequency": dict(sorted(label_frequency.items(), key=lambda x: x[0])),
        "labels_per_bundle_distribution": {
            str(k): v for k, v in sorted(label_count_dist.items(), key=lambda x: x[0])
        },
        "pct_bundles_image_download_ok": round(pct_download_ok, 4),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Preprocess bundle/product datasets for multi-label training.")
    parser.add_argument(
        "--bundles_csv",
        type=Path,
        default=Path("data/bundles_dataset.csv"),
        help="Path to bundles CSV (default: data/bundles_dataset.csv)",
    )
    parser.add_argument(
        "--train_relations_csv",
        type=Path,
        default=Path("data/bundles_product_match_train.csv"),
        help="Path to train relations CSV (default: data/bundles_product_match_train.csv)",
    )
    parser.add_argument(
        "--products_csv",
        type=Path,
        default=Path("data/product_dataset.csv"),
        help="Path to products CSV (default: data/product_dataset.csv)",
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path("data/bundles_product_match_test.csv"),
        help="Optional path to test CSV (default: data/bundles_product_match_test.csv)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Output directory (default: data/preprocessed)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download_products", action="store_true")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def validate_test_csv(
    test_df: pd.DataFrame,
    bundles_df: pd.DataFrame,
    strict: bool,
) -> Dict[str, int]:
    """Validate test bundle ids against bundles catalog."""
    test_df = test_df.copy()
    test_df["bundle_asset_id"] = test_df["bundle_asset_id"].astype(str)
    valid_bundle_ids = set(bundles_df["bundle_asset_id"].astype(str))
    missing_mask = ~test_df["bundle_asset_id"].isin(valid_bundle_ids)
    missing = int(missing_mask.sum())
    total = int(len(test_df))
    if missing:
        LOGGER.warning(
            "test_csv contains %d bundle_asset_id values absent in bundles_csv%s",
            missing,
            " (strict mode will fail)" if strict else "",
        )
    if strict and missing:
        raise ValueError(f"test_csv integrity error: missing bundles={missing}")
    return {"test_rows_total": total, "test_rows_missing_bundle": missing}


def main() -> None:
    """Run preprocessing pipeline."""
    configure_logging()
    args = parse_args()

    if args.max_workers <= 0:
        raise ValueError("--max_workers must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")

    bundles_df = read_csv_with_schema(
        args.bundles_csv,
        required_columns=["bundle_asset_id", "bundle_image_url"],
        name="bundles_csv",
    )
    relations_df = read_csv_with_schema(
        args.train_relations_csv,
        required_columns=["bundle_asset_id", "product_asset_id"],
        name="train_relations_csv",
    )
    products_df = read_csv_with_schema(
        args.products_csv,
        required_columns=["product_asset_id", "product_image_url", "product_description"],
        name="products_csv",
    )

    extra_counts: Dict[str, int] = {}
    if args.test_csv is not None:
        test_df = read_csv_with_schema(
            args.test_csv,
            required_columns=["bundle_asset_id", "product_asset_id"],
            name="test_csv",
        )
        extra_counts.update(validate_test_csv(test_df, bundles_df, strict=args.strict))

    records, integrity_counts = build_bundle_records(
        bundles_df=bundles_df,
        relations_df=relations_df,
        products_df=products_df,
        strict=args.strict,
        limit=args.limit,
    )
    if not records:
        raise RuntimeError("No valid bundle records were created. Check input data and cleaning rules.")

    label_vocab, label2idx = build_label_vocab(records)
    train_records, val_records = split_train_val(records, val_ratio=args.val_ratio, seed=args.seed)

    out_dir: Path = args.out_dir
    bundle_img_dir = out_dir / "images" / "bundles"
    product_img_dir = out_dir / "images" / "products"
    manifests_dir = out_dir / "manifests"
    ensure_dir(bundle_img_dir)
    ensure_dir(manifests_dir)
    if args.download_products:
        ensure_dir(product_img_dir)

    bundle_tasks = [
        (r.bundle_asset_id, bundle_img_dir / f"{r.bundle_asset_id}.jpg", r.image_url)
        for r in records
    ]
    LOGGER.info("Downloading bundle images (%d items)...", len(bundle_tasks))
    bundle_image_status, bundle_forbidden_urls = download_many_images(
        tasks=bundle_tasks,
        max_workers=args.max_workers,
        timeout=args.timeout,
        retries=args.retries,
    )

    if args.download_products:
        unique_products = products_df.drop_duplicates(subset=["product_asset_id"], keep="first")
        product_tasks: List[Tuple[str, Path, str]] = []
        for row in unique_products.itertuples(index=False):
            product_id = str(row.product_asset_id)
            product_url = "" if pd.isna(row.product_image_url) else str(row.product_image_url).strip()
            out_path = product_img_dir / f"{product_id}.jpg"
            product_tasks.append((product_id, out_path, product_url))
        LOGGER.info("Downloading product images (%d items)...", len(product_tasks))
        _, product_forbidden_urls = download_many_images(
            tasks=product_tasks,
            max_workers=args.max_workers,
            timeout=args.timeout,
            retries=args.retries,
        )
    else:
        product_forbidden_urls = []

    forbidden_urls = sorted(set(bundle_forbidden_urls + product_forbidden_urls))
    if forbidden_urls:
        forbidden_path = out_dir / "download_failures_forbidden.txt"
        with forbidden_path.open("w", encoding="utf-8") as fh:
            for url in forbidden_urls:
                fh.write(url + "\n")
        LOGGER.warning("Saved %d forbidden URLs to %s", len(forbidden_urls), forbidden_path.resolve())

    write_manifest(
        manifests_dir / "train_manifest.jsonl",
        train_records,
        label2idx,
        bundle_image_status,
        bundle_img_dir,
    )
    write_manifest(
        manifests_dir / "val_manifest.jsonl",
        val_records,
        label2idx,
        bundle_image_status,
        bundle_img_dir,
    )

    labels_path = out_dir / "labels.json"
    label2idx_path = out_dir / "label2idx.json"
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump(label_vocab, fh, ensure_ascii=False, indent=2)
    with label2idx_path.open("w", encoding="utf-8") as fh:
        json.dump(label2idx, fh, ensure_ascii=False, indent=2)

    stats = compute_stats(
        records=records,
        train_records=train_records,
        val_records=val_records,
        label_vocab=label_vocab,
        image_status=bundle_image_status,
    )
    stats["integrity"] = {**integrity_counts, **extra_counts}
    with (out_dir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    LOGGER.info("Done. Output written to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
