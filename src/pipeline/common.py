from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import read_jsonl


def load_manifest_map(
    manifest_path: str | Path,
    key_field: str,
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    return {str(row[key_field]): row for row in rows}
