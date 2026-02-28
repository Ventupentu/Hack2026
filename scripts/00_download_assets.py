#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, setup_logging
from src.utils.image import load_image_rgb

LOGGER = logging.getLogger("download_assets")


@dataclass(frozen=True)
class DownloadTask:
    asset_id: str
    url: str
    out_path: Path


def build_session(retries: int, backoff_factor: float = 0.5) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        backoff_factor=backoff_factor,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def infer_extension(url: str) -> str:
    path = urlparse(url).path.lower()
    ext = os.path.splitext(path)[1]
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return ext
    return ".jpg"


def resolve_default_path(preferred: str, alternatives: list[str]) -> Path:
    candidate = ROOT / preferred
    if candidate.exists():
        return candidate
    for alt in alternatives:
        p = ROOT / alt
        if p.exists():
            return p
    return candidate


def read_tasks(csv_path: Path, id_col: str, url_col: str, out_dir: Path) -> list[DownloadTask]:
    df = pd.read_csv(csv_path)
    df = df[[id_col, url_col]].dropna().drop_duplicates(subset=[id_col])

    tasks: list[DownloadTask] = []
    for row in df.itertuples(index=False):
        asset_id = str(getattr(row, id_col))
        url = str(getattr(row, url_col)).strip()
        ext = infer_extension(url)
        out_path = out_dir / f"{asset_id}{ext}"
        tasks.append(DownloadTask(asset_id=asset_id, url=url, out_path=out_path))
    return tasks


def download_one(
    task: DownloadTask,
    session: requests.Session,
    timeout: float,
    skip_existing: bool,
    validate_image: bool,
) -> tuple[str, str, bool, str]:
    """Return: (asset_id, out_path, success, error_msg)."""
    out_path = task.out_path

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        if not validate_image or load_image_rgb(out_path) is not None:
            return task.asset_id, str(out_path), True, "cached"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        with session.get(task.url, timeout=timeout, stream=True) as resp:
            if resp.status_code >= 400:
                return task.asset_id, str(out_path), False, f"http_{resp.status_code}"
            with tmp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp_path.replace(out_path)

        if validate_image and load_image_rgb(out_path) is None:
            out_path.unlink(missing_ok=True)
            return task.asset_id, str(out_path), False, "invalid_image"

        return task.asset_id, str(out_path), True, "downloaded"
    except Exception as exc:  # noqa: BLE001
        tmp_path.unlink(missing_ok=True)
        return task.asset_id, str(out_path), False, str(exc)


def run_downloads(
    tasks: list[DownloadTask],
    workers: int,
    retries: int,
    timeout: float,
    skip_existing: bool,
    validate_image: bool,
) -> pd.DataFrame:
    session = build_session(retries=retries)
    rows = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, task, session, timeout, skip_existing, validate_image): task
            for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            asset_id, path, success, message = future.result()
            rows.append(
                {
                    "asset_id": asset_id,
                    "image_path": path if success else "",
                    "success": int(success),
                    "status": message,
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download bundle and product assets with retries and cache")
    parser.add_argument(
        "--bundles_csv",
        type=Path,
        default=resolve_default_path("data/bundles.csv", ["data/bundles_dataset.csv"]),
        help="CSV with bundle_asset_id and bundle_image_url",
    )
    parser.add_argument(
        "--products_csv",
        type=Path,
        default=resolve_default_path("data/products.csv", ["data/product_dataset.csv"]),
        help="CSV with product_asset_id and product_image_url",
    )
    parser.add_argument("--bundle_out_dir", type=Path, default=Path("data/bundle_images"))
    parser.add_argument("--product_out_dir", type=Path, default=Path("data/product_images"))
    parser.add_argument("--bundle_index_out", type=Path, default=Path("artifacts/paths/bundle_paths.csv"))
    parser.add_argument("--product_index_out", type=Path, default=Path("artifacts/paths/product_paths.csv"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--validate_images", action="store_true", default=True)
    parser.add_argument("--no_validate_images", action="store_false", dest="validate_images")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    LOGGER.info("Reading bundles CSV: %s", args.bundles_csv)
    bundle_tasks = read_tasks(
        csv_path=args.bundles_csv,
        id_col="bundle_asset_id",
        url_col="bundle_image_url",
        out_dir=ensure_dir(args.bundle_out_dir),
    )
    LOGGER.info("Reading products CSV: %s", args.products_csv)
    product_tasks = read_tasks(
        csv_path=args.products_csv,
        id_col="product_asset_id",
        url_col="product_image_url",
        out_dir=ensure_dir(args.product_out_dir),
    )

    LOGGER.info("Downloading %d bundle images", len(bundle_tasks))
    bundle_df = run_downloads(
        tasks=bundle_tasks,
        workers=args.workers,
        retries=args.retries,
        timeout=args.timeout,
        skip_existing=args.skip_existing,
        validate_image=args.validate_images,
    ).rename(columns={"asset_id": "bundle_asset_id"})

    LOGGER.info("Downloading %d product images", len(product_tasks))
    product_df = run_downloads(
        tasks=product_tasks,
        workers=args.workers,
        retries=args.retries,
        timeout=args.timeout,
        skip_existing=args.skip_existing,
        validate_image=args.validate_images,
    ).rename(columns={"asset_id": "product_asset_id"})

    ensure_dir(args.bundle_index_out.parent)
    ensure_dir(args.product_index_out.parent)
    bundle_df.to_csv(args.bundle_index_out, index=False)
    product_df.to_csv(args.product_index_out, index=False)

    LOGGER.info(
        "Bundle success: %d/%d",
        int(bundle_df["success"].sum()),
        len(bundle_df),
    )
    LOGGER.info(
        "Product success: %d/%d",
        int(product_df["success"].sum()),
        len(product_df),
    )
    LOGGER.info("Saved bundle paths: %s", args.bundle_index_out)
    LOGGER.info("Saved product paths: %s", args.product_index_out)


if __name__ == "__main__":
    main()
