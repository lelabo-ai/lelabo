from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .registry import get_capsule, remove_capsule_entry


def remove_capsule(
    *,
    capsule_or_alias: str,
    capsules_dir: Path | None = None,
    delete_files: bool = True,
) -> dict[str, Any]:
    row = get_capsule(capsule_or_alias, capsules_dir)
    if row is None:
        raise ValueError(f"Unknown capsule '{capsule_or_alias}'.")

    capsule_path = Path(str(row.get("path", ""))).resolve()

    deleted = False
    if delete_files and str(capsule_path):
        if capsule_path.is_dir():
            shutil.rmtree(capsule_path)
            deleted = True
        elif capsule_path.exists():
            capsule_path.unlink()
            deleted = True

    removed = remove_capsule_entry(capsule_or_alias, capsules_dir)
    out = dict(removed)
    out["deleted_files"] = bool(deleted)
    out["delete_files_requested"] = bool(delete_files)
    return out


__all__ = ["remove_capsule"]
