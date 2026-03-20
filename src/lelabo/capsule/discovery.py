"""Shared capsule discovery helpers for workspace and store-aware CLI flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .gitspace import find_gitspace_root, load_gitspace
from .install import inspect_capsule_directory
from .registry import default_capsules_dir, list_capsules


_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "venv",
    "build",
    "dist",
    "outputs",
}


@dataclass(frozen=True)
class DiscoveredCapsule:
    capsule_id: str
    root: Path
    path: str
    status: str
    source_kind: str
    aliases: tuple[str, ...] = ()
    gitspace_root: str | None = None
    from_store: bool = False
    from_registry: bool = False


def is_capsule_root(path: Path) -> bool:
    candidate = path.expanduser().resolve()
    return candidate.is_dir() and (
        (candidate / "capsule.toml").is_file() or (candidate / "manifest.json").is_file()
    )


def find_capsule_root(start: Path | None = None) -> Path | None:
    cursor = (start or Path.cwd()).expanduser().resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for current in (cursor, *cursor.parents):
        if is_capsule_root(current):
            return current
    return None


def current_workspace_root(start: Path | None = None) -> Path:
    cursor = (start or Path.cwd()).expanduser().resolve()
    gitspace_root = find_gitspace_root(cursor)
    if gitspace_root is not None:
        return gitspace_root.resolve()
    active = find_capsule_root(cursor)
    if active is not None:
        return active.resolve()
    return cursor


def _path_within(root: Path, candidate: Path) -> bool:
    resolved_root = root.expanduser().resolve()
    resolved_candidate = candidate.expanduser().resolve()
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def _skip_dir(path: Path) -> bool:
    name = path.name.strip()
    if not name:
        return False
    if name.startswith(".") and name != ".lelabo":
        return True
    return name in _SKIP_DIR_NAMES


def _scan_descendant_capsules(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root.expanduser().resolve()]
    while stack:
        current = stack.pop()
        if is_capsule_root(current):
            out.append(current)
            continue
        try:
            children = sorted(
                (child for child in current.iterdir() if child.is_dir() and not _skip_dir(child)),
                key=lambda item: item.name,
                reverse=True,
            )
        except OSError:
            continue
        stack.extend(children)
    return sorted(out)


def _capsule_id_for_root(root: Path) -> str:
    info = inspect_capsule_directory(root)
    token = str(info.get("capsule_id", "")).strip()
    return token or root.name


def _workspace_entry(
    root: Path,
    *,
    source_kind: str,
    capsule_id: str | None = None,
    aliases: tuple[str, ...] = (),
    gitspace_root: Path | None = None,
    from_registry: bool = False,
) -> DiscoveredCapsule:
    resolved = root.expanduser().resolve()
    return DiscoveredCapsule(
        capsule_id=str(capsule_id or _capsule_id_for_root(resolved)).strip() or resolved.name,
        root=resolved,
        path=str(resolved),
        status="workspace",
        source_kind=source_kind,
        aliases=tuple(sorted({str(item).strip() for item in aliases if str(item).strip()})),
        gitspace_root=str(gitspace_root.resolve()) if gitspace_root is not None else None,
        from_store=False,
        from_registry=bool(from_registry),
    )


def _store_entry(root: Path, *, capsule_id: str, aliases: tuple[str, ...], source_kind: str) -> DiscoveredCapsule:
    resolved = root.expanduser().resolve()
    return DiscoveredCapsule(
        capsule_id=str(capsule_id).strip() or resolved.name,
        root=resolved,
        path=str(resolved),
        status="store",
        source_kind=source_kind,
        aliases=tuple(sorted({str(item).strip() for item in aliases if str(item).strip()})),
        gitspace_root=None,
        from_store=True,
        from_registry=True,
    )


def discover_workspace_capsules(start: Path | None = None) -> tuple[DiscoveredCapsule, ...]:
    cursor = (start or Path.cwd()).expanduser().resolve()
    gitspace_root = find_gitspace_root(cursor)
    if gitspace_root is not None:
        try:
            gitspace = load_gitspace(gitspace_root)
        except Exception:
            gitspace = None
        if gitspace is not None:
            out: list[DiscoveredCapsule] = []
            for item in list(gitspace.get("capsules", []) or []):
                rel_path = Path(str(item.get("path", "")).strip())
                capsule_root = (gitspace_root / rel_path).resolve()
                if not is_capsule_root(capsule_root):
                    continue
                out.append(
                    _workspace_entry(
                        capsule_root,
                        source_kind="gitspace_manifest",
                        capsule_id=str(item.get("id", "")).strip() or None,
                        gitspace_root=gitspace_root,
                    )
                )
            if out:
                return tuple(sorted(out, key=lambda item: (item.capsule_id, item.path)))

    active = find_capsule_root(cursor)
    if active is not None:
        return (_workspace_entry(active, source_kind="active"),)

    rows = [_workspace_entry(root, source_kind="workspace_descendant") for root in _scan_descendant_capsules(cursor)]
    return tuple(rows)


def discover_visible_capsules(
    *,
    start: Path | None = None,
    capsules_dir: Path | None = None,
) -> tuple[DiscoveredCapsule, ...]:
    merged: dict[str, DiscoveredCapsule] = {}

    def _merge(entry: DiscoveredCapsule) -> None:
        key = str(entry.root.resolve())
        existing = merged.get(key)
        if existing is None:
            merged[key] = entry
            return
        aliases = tuple(sorted({*existing.aliases, *entry.aliases}))
        status = existing.status
        source_kind = existing.source_kind
        from_store = existing.from_store or entry.from_store
        from_registry = existing.from_registry or entry.from_registry
        gitspace_root = existing.gitspace_root or entry.gitspace_root
        if existing.status != "workspace" and entry.status == "workspace":
            status = entry.status
            source_kind = entry.source_kind
        merged[key] = DiscoveredCapsule(
            capsule_id=existing.capsule_id or entry.capsule_id,
            root=existing.root,
            path=existing.path,
            status=status,
            source_kind=source_kind,
            aliases=aliases,
            gitspace_root=gitspace_root,
            from_store=from_store,
            from_registry=from_registry,
        )

    for entry in discover_workspace_capsules(start=start):
        _merge(entry)

    store_root = (capsules_dir or default_capsules_dir()).expanduser().resolve()
    for row in list_capsules(capsules_dir):
        raw_path = str(row.get("path", "")).strip()
        if not raw_path:
            continue
        root = Path(raw_path).expanduser().resolve()
        aliases = tuple(str(item).strip() for item in list(row.get("aliases", []) or []) if str(item).strip())
        capsule_id = str(row.get("capsule_id", "")).strip() or root.name
        if _path_within(store_root, root):
            _merge(_store_entry(root, capsule_id=capsule_id, aliases=aliases, source_kind="store_registry"))
            continue
        _merge(
            _workspace_entry(
                root,
                source_kind="attached_registry",
                capsule_id=capsule_id,
                aliases=aliases,
                from_registry=True,
            )
        )

    rows = sorted(
        merged.values(),
        key=lambda item: (0 if item.status == "workspace" else 1, item.capsule_id, item.path),
    )
    return tuple(rows)


__all__ = [
    "DiscoveredCapsule",
    "current_workspace_root",
    "discover_visible_capsules",
    "discover_workspace_capsules",
    "find_capsule_root",
    "is_capsule_root",
]
