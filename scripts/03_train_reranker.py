#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, setup_logging

LOGGER = logging.getLogger("reranker_disabled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrained-only mode: reranker training is disabled",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/reranker"))
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    ensure_dir(args.output_dir)

    report = {
        "mode": "pretrained_only",
        "status": "skipped",
        "reason": "No reranker fine-tuning in pretrained-only pipeline.",
        "instruction": "Run inference/eval without --use_reranker.",
    }

    report_path = args.output_dir / "reranker_status.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info("Reranker training skipped (pretrained-only mode)")
    LOGGER.info("Status file: %s", report_path)


if __name__ == "__main__":
    main()
