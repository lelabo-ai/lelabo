from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .registry import default_capsules_dir, get_capsule, remove_capsule_entry


def remove_capsule(
    *,
    capsule_or_alias: str,
    capsules_dir: Path | None = None,
    delete_files: bool = True,
    allow_external_delete: bool = False,
) -> dict[str, Any]:
    row = get_capsule(capsule_or_alias, capsules_dir)
    if row is None:
        raise ValueError(f"Unknown capsule '{capsule_or_alias}'.")

    root = (capsules_dir or default_capsules_dir()).resolve()
    capsule_path = Path(str(row.get("path", ""))).resolve()

    deleted = False
    if delete_files and str(capsule_path):
        if capsule_path == root:
            raise ValueError(
                f"Refusing to delete capsules store root '{root}'."
            )
        is_outside_store = root not in capsule_path.parents
        if is_outside_store and not allow_external_delete:
            raise ValueError(
                f"Refusing to delete capsule path outside capsules store '{root}': {capsule_path}. "
                "Use '-rf' (or '--force-external-delete') to allow external deletion."
            )
        if allow_external_delete and capsule_path == Path(capsule_path.anchor):
            raise ValueError(f"Refusing to delete filesystem root path: {capsule_path}")
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
    out["allow_external_delete"] = bool(allow_external_delete)
    return out


__all__ = ["remove_capsule"]
