from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .registry import get_capsule
from .schema import normalize_capsule_id


def _safe_restore_dst(root: Path, name: str) -> Path:
    dst = (root / name).resolve()
    if dst == root or root not in dst.parents:
        raise ValueError(f"Unsafe restore destination path resolved outside target directory: {dst}")
    return dst


def restore_capsule(
    *,
    capsule_or_alias: str,
    destination_dir: Path | None = None,
    name: str | None = None,
    capsules_dir: Path | None = None,
) -> dict[str, Any]:
    row = get_capsule(capsule_or_alias, capsules_dir)
    if row is None:
        raise ValueError(f"Unknown capsule '{capsule_or_alias}'.")

    source_path = Path(str(row.get("path", ""))).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Capsule files not found on disk: {source_path}")

    root = (destination_dir or Path.cwd()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder_name = normalize_capsule_id(name or str(row.get("capsule_id", "")), field_name="name")
    dst = _safe_restore_dst(root, folder_name)
    if dst.exists():
        raise FileExistsError(f"Cannot restore capsule: destination already exists: {dst}")

    shutil.copytree(source_path, dst)

    return {
        "capsule_id": str(row.get("capsule_id", "")),
        "requested": str(capsule_or_alias),
        "source_path": str(source_path),
        "restored_path": str(dst),
    }


__all__ = ["restore_capsule"]
