"""Backward-compatible wrapper for phase-1 inference entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "src").is_dir():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_ensure_project_root_on_path()

from src.workflows import inference_phase1 as _impl

# Re-export public and internal symbols for import compatibility.
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

if __name__ == "__main__":
    _impl.main()
