"""Gitspace manifest helpers for multi-capsule Git repositories."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from .install import inspect_capsule_directory


GITSPACE_MANIFEST_RELATIVE_PATH = Path(".lelabo") / "gitspace.toml"


def gitspace_manifest_path(root: Path) -> Path:
    """Return the manifest path for one gitspace root."""
    return root.resolve() / GITSPACE_MANIFEST_RELATIVE_PATH


def find_gitspace_root(start: Path | None = None) -> Path | None:
    """Find the nearest gitspace root by searching upward for `.lelabo/gitspace.toml`."""
    cursor = (start or Path.cwd()).expanduser().resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for current in [cursor, *cursor.parents]:
        if gitspace_manifest_path(current).is_file():
            return current.resolve()
    return None


def _safe_rel_path(root: Path, target: Path) -> str:
    rel = target.resolve().relative_to(root.resolve())
    token = rel.as_posix()
    if not token or token == ".":
        raise ValueError("Gitspace capsule path cannot point to the gitspace root itself.")
    return token


def _capsule_entries_from_raw(root: Path, raw_capsules: Any) -> list[dict[str, str]]:
    if raw_capsules is None:
        return []
    if not isinstance(raw_capsules, list):
        raise ValueError("Invalid gitspace manifest: 'capsules' must be a list of tables.")

    out: list[dict[str, str]] = []
    ids_seen: set[str] = set()
    paths_seen: set[str] = set()
    for item in raw_capsules:
        if not isinstance(item, dict):
            raise ValueError("Invalid gitspace manifest: each [[capsules]] entry must be a table.")
        capsule_id = str(item.get("id", "")).strip()
        capsule_path = str(item.get("path", "")).strip()
        if not capsule_id:
            raise ValueError("Invalid gitspace manifest: each capsule entry needs a non-empty 'id'.")
        if not capsule_path:
            raise ValueError(f"Invalid gitspace manifest: capsule '{capsule_id}' needs a non-empty 'path'.")
        if capsule_id in ids_seen:
            raise ValueError(f"Invalid gitspace manifest: duplicate capsule id '{capsule_id}'.")
        if capsule_path in paths_seen:
            raise ValueError(f"Invalid gitspace manifest: duplicate capsule path '{capsule_path}'.")

        capsule_root = (root / capsule_path).resolve()
        if root.resolve() not in capsule_root.parents:
            raise ValueError(
                f"Invalid gitspace manifest: capsule path '{capsule_path}' escapes the gitspace root."
            )
        if not capsule_root.exists() or not capsule_root.is_dir():
            raise ValueError(
                f"Invalid gitspace manifest: capsule path '{capsule_path}' does not exist under '{root}'."
            )
        inspect_capsule_directory(capsule_root)
        ids_seen.add(capsule_id)
        paths_seen.add(capsule_path)
        out.append({"id": capsule_id, "path": capsule_path})
    return out


def load_gitspace(root_or_path: Path) -> dict[str, Any]:
    """Load and validate one gitspace manifest."""
    root = root_or_path.expanduser().resolve()
    manifest_path = gitspace_manifest_path(root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Gitspace manifest not found: {manifest_path}")

    raw = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid gitspace manifest: expected TOML table at '{manifest_path}'.")
    gitspace_section = raw.get("gitspace", {})
    if not isinstance(gitspace_section, dict):
        raise ValueError("Invalid gitspace manifest: [gitspace] table is required.")

    name = str(gitspace_section.get("name", "")).strip() or root.name
    capsules = _capsule_entries_from_raw(root, raw.get("capsules"))
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "name": name,
        "capsules": capsules,
    }


def init_gitspace(root: Path, *, name: str | None = None) -> dict[str, Any]:
    """Create a gitspace manifest when missing and return the loaded gitspace."""
    gitspace_root = root.expanduser().resolve()
    manifest_path = gitspace_manifest_path(gitspace_root)
    gitspace_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        token = str(name or gitspace_root.name).strip() or gitspace_root.name or "gitspace"
        safe_name = token.replace("\\", "\\\\").replace('"', '\\"')
        manifest_path.write_text(
            "\n".join(
                [
                    "[gitspace]",
                    f'name = "{safe_name}"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return load_gitspace(gitspace_root)


def dumps_gitspace_manifest(name: str, capsules: list[dict[str, str]]) -> str:
    """Serialize a gitspace manifest to TOML."""
    safe_name = str(name).replace("\\", "\\\\").replace('"', '\\"')
    lines = ["[gitspace]", f'name = "{safe_name}"', ""]
    for capsule in capsules:
        cid = str(capsule["id"]).replace("\\", "\\\\").replace('"', '\\"')
        path = str(capsule["path"]).replace("\\", "\\\\").replace('"', '\\"')
        lines.extend(
            [
                "[[capsules]]",
                f'id = "{cid}"',
                f'path = "{path}"',
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def add_capsule_to_gitspace(
    capsule_path: Path,
    *,
    gitspace_root: Path,
    capsule_id: str | None = None,
) -> dict[str, Any]:
    """Register a capsule path in one gitspace manifest without moving files."""
    root = gitspace_root.expanduser().resolve()
    gitspace = init_gitspace(root)
    capsule_root = capsule_path.expanduser().resolve()
    if not capsule_root.exists() or not capsule_root.is_dir():
        raise FileNotFoundError(f"Capsule source directory not found: {capsule_root}")
    rel_path = _safe_rel_path(root, capsule_root)
    capsule_info = inspect_capsule_directory(capsule_root)
    resolved_id = str(capsule_id or capsule_info["capsule_id"]).strip()

    entries = list(gitspace["capsules"])
    for item in entries:
        if item["id"] == resolved_id:
            if item["path"] == rel_path:
                out = dict(gitspace)
                out["action"] = "unchanged"
                out["capsule"] = {"id": resolved_id, "path": rel_path}
                return out
            raise ValueError(
                f"Gitspace '{root}' already contains capsule id '{resolved_id}' at '{item['path']}'."
            )
        if item["path"] == rel_path:
            raise ValueError(
                f"Gitspace '{root}' already contains capsule path '{rel_path}' as id '{item['id']}'."
            )

    entries.append({"id": resolved_id, "path": rel_path})
    manifest_path = gitspace_manifest_path(root)
    manifest_path.write_text(dumps_gitspace_manifest(gitspace["name"], entries), encoding="utf-8")
    out = load_gitspace(root)
    out["action"] = "added"
    out["capsule"] = {"id": resolved_id, "path": rel_path}
    return out


def resolve_gitspace_capsule(gitspace: dict[str, Any], capsule_ref: str) -> dict[str, str]:
    """Resolve one capsule entry from a gitspace by id."""
    token = str(capsule_ref).strip()
    if not token:
        raise ValueError("Gitspace capsule selection cannot be empty.")
    for item in list(gitspace.get("capsules", []) or []):
        if str(item.get("id", "")).strip() == token:
            return {"id": str(item["id"]), "path": str(item["path"])}
    raise ValueError(
        f"Unknown gitspace capsule '{token}'. Available: "
        + ", ".join(str(item.get("id", "")) for item in list(gitspace.get("capsules", []) or []))
    )


__all__ = [
    "GITSPACE_MANIFEST_RELATIVE_PATH",
    "gitspace_manifest_path",
    "find_gitspace_root",
    "load_gitspace",
    "init_gitspace",
    "add_capsule_to_gitspace",
    "resolve_gitspace_capsule",
    "dumps_gitspace_manifest",
]
