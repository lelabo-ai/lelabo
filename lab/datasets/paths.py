from __future__ import annotations

import os
from pathlib import Path


def _default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / ".cache" / "data"


DATA_DIR = Path(os.environ.get("LELABO_DATA_DIR", _default_data_dir())).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)


def dataset_dir(name: str) -> Path:
    p = DATA_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p
