from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


INDEX_SCHEMA_VERSION = "1.0"


def _user_data_capsules_dir() -> Path:
    # Cross-platform app-data default, similar to how mature CLIs store local state.
    home = Path.home()
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(home / "AppData" / "Local")
        return (Path(base).expanduser().resolve() / "LeLabo" / "capsules").resolve()
    if sys.platform == "darwin":
        return (home / "Library" / "Application Support" / "lelabo" / "capsules").resolve()
    xdg = os.getenv("XDG_DATA_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser().resolve() / "lelabo" / "capsules").resolve()
    return (home / ".local" / "share" / "lelabo" / "capsules").resolve()


def default_capsules_dir() -> Path:
    from os import getenv

    raw = getenv("LELABO_CAPSULES_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()

    # Backward-compatible legacy project-local location.
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        maybe = candidate / ".lelabo" / "capsules"
        if maybe.exists():
            return maybe.resolve()

    return _user_data_capsules_dir()


def index_path(capsules_dir: Path | None = None) -> Path:
    root = (capsules_dir or default_capsules_dir()).resolve()
    return root / "index.json"


def _empty_index() -> Dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capsules": {},
        "aliases": {},
    }


def load_index(capsules_dir: Path | None = None) -> Dict[str, Any]:
    idx_path = index_path(capsules_dir)
    if not idx_path.exists():
        return _empty_index()

    data = json.loads(idx_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return _empty_index()

    data.setdefault("schema_version", INDEX_SCHEMA_VERSION)
    data.setdefault("updated_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    data.setdefault("capsules", {})
    data.setdefault("aliases", {})
    return data


def save_index(index: Dict[str, Any], capsules_dir: Path | None = None) -> Path:
    idx_path = index_path(capsules_dir)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    index["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return idx_path


def add_capsule_entry(
    *,
    capsule_id: str,
    capsule_path: Path,
    manifest: Dict[str, Any],
    alias: str | None = None,
    source_bundle: str | None = None,
    capsules_dir: Path | None = None,
) -> Dict[str, Any]:
    index = load_index(capsules_dir)

    entry = {
        "capsule_id": capsule_id,
        "path": str(capsule_path.resolve()),
        "manifest_path": str((capsule_path / "manifest.json").resolve()),
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_bundle": source_bundle,
        "kind": manifest.get("kind"),
        "created_at": manifest.get("created_at"),
        "source": manifest.get("source", {}),
    }
    index["capsules"][capsule_id] = entry
    if alias:
        index["aliases"][alias] = capsule_id

    save_index(index, capsules_dir)
    return entry


def resolve_capsule_id(capsule_or_alias: str, capsules_dir: Path | None = None) -> Optional[str]:
    index = load_index(capsules_dir)
    if capsule_or_alias in index["capsules"]:
        return capsule_or_alias
    return index.get("aliases", {}).get(capsule_or_alias)


def get_capsule(capsule_or_alias: str, capsules_dir: Path | None = None) -> Optional[Dict[str, Any]]:
    cid = resolve_capsule_id(capsule_or_alias, capsules_dir)
    if cid is None:
        return None
    index = load_index(capsules_dir)
    return index["capsules"].get(cid)


def list_capsules(capsules_dir: Path | None = None) -> list[Dict[str, Any]]:
    index = load_index(capsules_dir)
    out: list[Dict[str, Any]] = []
    for cid, entry in index.get("capsules", {}).items():
        aliases = [a for a, val in index.get("aliases", {}).items() if val == cid]
        row = dict(entry)
        row["aliases"] = sorted(aliases)
        out.append(row)
    out.sort(key=lambda r: str(r.get("installed_at", "")), reverse=True)
    return out


def remove_capsule_entry(capsule_or_alias: str, capsules_dir: Path | None = None) -> Dict[str, Any]:
    index = load_index(capsules_dir)
    capsule_id = str(capsule_or_alias).strip()

    if capsule_id in index.get("capsules", {}):
        resolved_id = capsule_id
    else:
        resolved_id = index.get("aliases", {}).get(capsule_id)
        if not resolved_id:
            raise ValueError(f"Unknown capsule '{capsule_or_alias}'.")

    entry = index.get("capsules", {}).pop(resolved_id, None)
    if entry is None:
        raise ValueError(f"Unknown capsule '{capsule_or_alias}'.")

    removed_aliases = [a for a, v in list(index.get("aliases", {}).items()) if v == resolved_id]
    for alias in removed_aliases:
        index["aliases"].pop(alias, None)

    save_index(index, capsules_dir)
    out = dict(entry)
    out["aliases"] = sorted(removed_aliases)
    return out
