from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from lab.core.registry import Registry
from ...core.utils.capsule_plugins import (
    load_capsule_plugins,
    load_installed_capsule_plugins,
    reset_capsule_plugin_cache,
)

DATASET_REGISTRY = Registry("datasets", package="lab.supervised.datasets")
_BASE_DATASET_ITEMS: dict[str, Any] | None = None


def register_dataset(name: str):
    return DATASET_REGISTRY.register(name)


def _ensure_dataset_baseline() -> None:
    global _BASE_DATASET_ITEMS
    if _BASE_DATASET_ITEMS is not None:
        return
    _BASE_DATASET_ITEMS = DATASET_REGISTRY.snapshot_discovered_items()


def _refresh_dataset_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    _ensure_dataset_baseline()
    DATASET_REGISTRY._items = dict(_BASE_DATASET_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("datasets",))
    for root in list(extra_capsule_roots or []):
        load_capsule_plugins(kinds=("datasets",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("datasets",), capsules_dir=capsules_dir)


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
