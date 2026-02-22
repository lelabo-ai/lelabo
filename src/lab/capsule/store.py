from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..core.utils.capsule_plugins import find_active_capsule_root
from .registry import add_capsule_entry, default_capsules_dir
from .schema import normalize_capsule_id, validate_manifest


def _safe_capsule_dst(root: Path, capsule_id: str) -> Path:
    dst = (root / capsule_id).resolve()
    if dst == root or root not in dst.parents:
        raise ValueError(f"Unsafe capsule destination path resolved outside capsules dir: {dst}")
    return dst


def _resolve_source_root(source_path: Path | None) -> Path:
    if source_path is not None:
        start = source_path.expanduser().resolve()
        if not start.exists():
            raise FileNotFoundError(f"Capsule source path not found: {start}")
    else:
        start = None

    root = find_active_capsule_root(start=start)
    if root is None:
        if source_path is None:
            raise ValueError("No active capsule found from current directory. Use --from <capsule_path>.")
        raise ValueError(f"Path '{source_path}' is not inside a capsule (missing capsule.toml).")
    return root.resolve()


def store_capsule(
    *,
    alias: str | None = None,
    source_path: Path | None = None,
    capsules_dir: Path | None = None,
) -> dict[str, Any]:
    source_root = _resolve_source_root(source_path)
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"manifest.json not found in capsule root: {source_root}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest.json in '{source_root}': {exc}") from exc

    validate_manifest(manifest)
    capsule_id = normalize_capsule_id(manifest.get("capsule_id", ""), field_name="manifest.capsule_id")

    root = (capsules_dir or default_capsules_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dst = _safe_capsule_dst(root, capsule_id)

    if source_root != dst:
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.copytree(source_root, dst)

    entry = add_capsule_entry(
        capsule_id=capsule_id,
        capsule_path=dst,
        manifest=manifest,
        alias=alias,
        source_bundle=f"local_path:{source_root}",
        capsules_dir=root,
    )
    out = dict(entry)
    out["stored_from"] = str(source_root)
    return out


__all__ = ["store_capsule"]
