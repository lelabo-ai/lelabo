from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from lab.core.registry import Registry
from ...capsule.registry import index_path
from ...core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    plugin_files_fingerprint,
    reset_capsule_plugin_cache,
)

DATASET_REGISTRY = Registry("datasets", package="lab.supervised.datasets")
_BASE_DATASET_ITEMS: dict[str, Any] | None = None
_LAST_DATASET_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_DATASET_ITEMS: dict[str, Any] | None = None


def register_dataset(name: str):
    return DATASET_REGISTRY.register(name)


def _ensure_dataset_baseline() -> None:
    global _BASE_DATASET_ITEMS
    if _BASE_DATASET_ITEMS is not None:
        return
    _BASE_DATASET_ITEMS = DATASET_REGISTRY.snapshot_discovered_items()


def _normalize_extra_roots(extra_capsule_roots: Sequence[Path] | None) -> tuple[Path, ...]:
    roots = {Path(root).resolve() for root in list(extra_capsule_roots or [])}
    return tuple(sorted(roots, key=str))


def _capsules_index_mtime_ns(capsules_dir: Path | None) -> int | None:
    idx = index_path(capsules_dir)
    try:
        return int(idx.stat().st_mtime_ns)
    except OSError:
        return None


def _build_refresh_key(
    *,
    capsules_dir: Path | None,
    extra_capsule_roots: Sequence[Path] | None,
) -> tuple[Any, ...]:
    active_root = find_active_capsule_root()
    normalized_extra = _normalize_extra_roots(extra_capsule_roots)
    strict_plugins = os.getenv("LELABO_STRICT_PLUGINS", "").strip().lower()
    return (
        str(index_path(capsules_dir).parent.resolve()),
        _capsules_index_mtime_ns(capsules_dir),
        str(active_root) if active_root is not None else None,
        plugin_files_fingerprint(active_root, kinds=("datasets",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("datasets",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_dataset_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_DATASET_REFRESH_KEY, _LAST_DATASET_ITEMS
    _ensure_dataset_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_DATASET_REFRESH_KEY == refresh_key and _LAST_DATASET_ITEMS is not None:
        DATASET_REGISTRY._items = dict(_LAST_DATASET_ITEMS)
        return

    DATASET_REGISTRY._items = dict(_BASE_DATASET_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("datasets",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("datasets",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("datasets",), capsules_dir=capsules_dir)
    _LAST_DATASET_REFRESH_KEY = refresh_key
    _LAST_DATASET_ITEMS = dict(DATASET_REGISTRY._items)


def get_dataset(name: str, **kwargs):
    _refresh_dataset_registry()
    builder = DATASET_REGISTRY.get(name)
    return builder(**kwargs)


def get_dataset_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
):
    _refresh_dataset_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return DATASET_REGISTRY.names()
