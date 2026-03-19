"""Workspace-first discovery of capsules and named sweep configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..capsule.plugins.discovery import find_active_capsule_root


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
class DiscoveredSweep:
    capsule_id: str
    capsule_root: Path
    sweep_name: str
    config_path: Path


@dataclass(frozen=True)
class DiscoveredCapsuleSweeps:
    capsule_id: str
    capsule_root: Path
    sweeps: tuple[DiscoveredSweep, ...]


def _is_capsule_root(path: Path) -> bool:
    return path.is_dir() and ((path / "capsule.toml").is_file() or (path / "manifest.json").is_file())


def _skip_dir(path: Path) -> bool:
    name = path.name.strip()
    if not name:
        return False
    if name.startswith(".") and name != ".lelabo":
        return True
    return name in _SKIP_DIR_NAMES


def _scan_descendant_capsules(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root.resolve()]
    while stack:
        current = stack.pop()
        if _is_capsule_root(current):
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


def _discovered_sweeps_for_capsule(capsule_root: Path) -> tuple[DiscoveredSweep, ...]:
    sweeps_dir = capsule_root / "sweeps"
    if not sweeps_dir.is_dir():
        return ()
    capsule_id = capsule_root.name
    items: list[DiscoveredSweep] = []
    for path in sorted(sweeps_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        items.append(
            DiscoveredSweep(
                capsule_id=capsule_id,
                capsule_root=capsule_root.resolve(),
                sweep_name=path.stem,
                config_path=path.resolve(),
            )
        )
    return tuple(items)


def discover_workspace_capsules(start: Path | None = None) -> tuple[DiscoveredCapsuleSweeps, ...]:
    """Discover descendant capsules and their named sweeps from the current workspace."""
    cwd = (start or Path.cwd()).expanduser().resolve()
    active_capsule = find_active_capsule_root(start=cwd)
    if active_capsule is not None:
        capsule_root = active_capsule.resolve()
        return (
            DiscoveredCapsuleSweeps(
                capsule_id=capsule_root.name,
                capsule_root=capsule_root,
                sweeps=_discovered_sweeps_for_capsule(capsule_root),
            ),
        )

    rows = []
    for capsule_root in _scan_descendant_capsules(cwd):
        rows.append(
            DiscoveredCapsuleSweeps(
                capsule_id=capsule_root.name,
                capsule_root=capsule_root,
                sweeps=_discovered_sweeps_for_capsule(capsule_root),
            )
        )
    return tuple(rows)


def resolve_discovered_sweep(
    capsules: tuple[DiscoveredCapsuleSweeps, ...],
    *,
    capsule_id: str | None,
    sweep_name: str,
) -> DiscoveredSweep:
    """Resolve one named sweep from discovered workspace capsules."""
    token = str(sweep_name).strip()
    if not token:
        raise ValueError("Sweep name cannot be empty.")

    candidates: list[DiscoveredSweep] = []
    for capsule in capsules:
        if capsule_id and capsule.capsule_id != str(capsule_id).strip():
            continue
        for sweep in capsule.sweeps:
            if sweep.sweep_name == token:
                candidates.append(sweep)

    if not candidates:
        if capsule_id:
            raise ValueError(f"Unknown sweep '{token}' in capsule '{capsule_id}'.")
        raise ValueError(f"Unknown sweep '{token}' in the current workspace.")
    if len(candidates) > 1:
        raise ValueError(
            f"Sweep '{token}' exists in multiple capsules. Re-run with `--capsule <capsule_id>`."
        )
    return candidates[0]


__all__ = [
    "DiscoveredCapsuleSweeps",
    "DiscoveredSweep",
    "discover_workspace_capsules",
    "resolve_discovered_sweep",
]
