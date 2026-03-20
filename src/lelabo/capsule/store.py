"""Stash helpers that move local capsules into the capsule store."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .discovery import current_workspace_root, is_capsule_root
from .install import compute_capsule_fingerprint
from .plugins.discovery import find_active_capsule_root
from .registry import add_capsule_entry, default_capsules_dir, get_capsule
from .schema import normalize_capsule_id, validate_manifest


def _safe_capsule_dst(root: Path, capsule_id: str) -> Path:
    dst = (root / capsule_id).resolve()
    if dst == root or root not in dst.parents:
        raise ValueError(f"Unsafe capsule destination path resolved outside capsules dir: {dst}")
    return dst


def _looks_like_capsule_root(path: Path) -> bool:
    return is_capsule_root(path)


def _path_within(root: Path, candidate: Path) -> bool:
    resolved_root = root.expanduser().resolve()
    resolved_candidate = candidate.expanduser().resolve()
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def _resolve_named_child(start: Path, alias: str | None) -> Path | None:
    if not alias:
        return None
    name = str(alias).strip()
    if not name:
        return None

    candidate = (start / name).resolve()
    if candidate == start or start not in candidate.parents:
        return None
    if not candidate.is_dir():
        return None
    if _looks_like_capsule_root(candidate):
        return candidate
    root = find_active_capsule_root(start=candidate)
    if root is not None and root.resolve() == candidate:
        return candidate
    return None


def _resolve_source_root(source_path: Path | None, alias: str | None) -> Path:
    if source_path is None:
        start = Path.cwd().resolve()
    else:
        start = source_path.expanduser().resolve()
        if not start.exists():
            raise FileNotFoundError(f"Capsule source path not found: {start}")

    if _looks_like_capsule_root(start):
        return start

    root = find_active_capsule_root(start=start)
    if root is not None:
        return root.resolve()

    named_child = _resolve_named_child(start, alias)
    if named_child is not None:
        return named_child

    if source_path is None:
        raise ValueError("No active capsule found from current directory. Use --from <capsule_path>.")

    name_hint = str(alias).strip() if alias else ""
    if name_hint:
        raise ValueError(
            f"Path '{source_path}' is not a capsule root (missing manifest.json/capsule.toml) "
            f"and does not contain a matching '{name_hint}' capsule folder."
        )
    raise ValueError(f"Path '{source_path}' is not a capsule root (missing manifest.json/capsule.toml).")


def stash_capsule(
    *,
    alias: str | None = None,
    source_path: Path | None = None,
    capsules_dir: Path | None = None,
) -> dict[str, Any]:
    source_root = _resolve_source_root(source_path, alias)
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

    moved = False
    if source_root != dst:
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(source_root), str(dst))
        moved = True

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
    out["moved"] = moved
    return out


def _existing_capsule_fingerprint(existing: dict[str, Any]) -> str | None:
    token = str(existing.get("fingerprint", "") or "").strip()
    if token:
        return token
    path = Path(str(existing.get("path", "") or "")).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return None
    try:
        return compute_capsule_fingerprint(path)
    except Exception:
        return None


def attach_capsule(
    *,
    alias: str | None = None,
    source_path: Path | None = None,
    capsules_dir: Path | None = None,
    capsule_id_override: str | None = None,
    force_replace: bool = False,
) -> dict[str, Any]:
    source_root = _resolve_source_root(source_path, alias)
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"manifest.json not found in capsule root: {source_root}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest.json in '{source_root}': {exc}") from exc

    validate_manifest(manifest)
    capsule_id = normalize_capsule_id(
        capsule_id_override or manifest.get("capsule_id", ""),
        field_name="manifest.capsule_id",
    )
    manifest["capsule_id"] = capsule_id
    manifest["fingerprint"] = compute_capsule_fingerprint(source_root)

    root = (capsules_dir or default_capsules_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if _path_within(root, source_root):
        raise ValueError(
            f"Capsule '{capsule_id}' already lives in the local store at '{source_root}'. "
            "Use `lelabo capsule checkout` to work on stored capsules, or `lelabo capsule stash` for the normal move-to-store flow."
        )

    workspace_root = current_workspace_root()
    if _path_within(workspace_root, source_root):
        raise ValueError(
            f"Capsule '{capsule_id}' is already in the current workspace at '{source_root}'. "
            "Use `lelabo capsule stash` to move it into the store. `attach` is only for external capsules."
        )

    existing = get_capsule(capsule_id, root)
    if existing is not None and not bool(force_replace):
        existing_path = Path(str(existing.get("path", "") or "")).expanduser().resolve()
        existing_fp = _existing_capsule_fingerprint(existing)
        if _path_within(root, existing_path):
            raise ValueError(
                f"Capsule id '{capsule_id}' already exists in the store at '{existing_path}'. "
                "Use `lelabo capsule checkout` or `lelabo capsule stash` as the normal store/workspace flow."
            )
        if existing_path == source_root.resolve() or (
            existing_fp and existing_fp == str(manifest.get("fingerprint", ""))
        ):
            out = dict(existing)
            out["attached_from"] = str(source_root)
            out["moved"] = False
            out["attach_action"] = "unchanged"
            out["replaced_existing"] = False
            return out
        raise ValueError(
            f"Capsule id '{capsule_id}' is already attached at '{existing.get('path')}'. "
            "Use `--force-replace` to replace the external registry entry, or `--rename-to <new_id>` to attach side-by-side."
        )

    entry = add_capsule_entry(
        capsule_id=capsule_id,
        capsule_path=source_root,
        manifest=manifest,
        alias=alias,
        source_bundle=f"local_path:{source_root}",
        capsules_dir=root,
    )
    out = dict(entry)
    out["attached_from"] = str(source_root)
    out["moved"] = False
    out["attach_action"] = "replaced" if bool(force_replace and existing is not None) else "attached"
    out["replaced_existing"] = bool(force_replace and existing is not None)
    return out


__all__ = ["stash_capsule", "attach_capsule"]
