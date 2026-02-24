from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .registry import default_capsules_dir, get_capsule, remove_capsule_entry
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

    cache_root = (capsules_dir or default_capsules_dir()).resolve()
    source_path = Path(str(row.get("path", ""))).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Capsule files not found on disk: {source_path}")
    if source_path == cache_root or cache_root not in source_path.parents:
        raise ValueError(
            f"Refusing to restore capsule files outside cache '{cache_root}': {source_path}"
        )

    root = (destination_dir or Path.cwd()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder_name = normalize_capsule_id(name or str(row.get("capsule_id", "")), field_name="name")
    dst = _safe_restore_dst(root, folder_name)
    if source_path == dst:
        raise ValueError(f"Restore destination is the same as cache source: {dst}")
    if source_path in dst.parents:
        raise ValueError(f"Restore destination cannot be inside cache source: {dst}")
    if dst.exists():
        raise FileExistsError(f"Cannot restore capsule: destination already exists: {dst}")

    shutil.move(str(source_path), str(dst))
    removed = remove_capsule_entry(capsule_or_alias, capsules_dir=cache_root)

    return {
        "capsule_id": str(row.get("capsule_id", "")),
        "requested": str(capsule_or_alias),
        "source_path": str(source_path),
        "restored_path": str(dst),
        "moved": True,
        "removed_from_cache": True,
        "removed_aliases": list(removed.get("aliases", [])),
    }


__all__ = ["restore_capsule"]
