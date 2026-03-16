"""Dataset registry and construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from lelabo.core.registry import Registry
from ...capsule.plugins.snapshot import build_capsule_registry_snapshot

DATASET_REGISTRY = Registry("datasets", package="lelabo.supervised.datasets")
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

def _dataset_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_dataset_baseline()
    return build_capsule_registry_snapshot(
        registry_name="datasets",
        kind="datasets",
        builtins=dict(_BASE_DATASET_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def get_dataset(name: str, **kwargs):
    builder = _dataset_snapshot().get(name)
    return builder(**kwargs)


def get_dataset_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
):
    return _dataset_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()
"""Dataset registry and construction helpers."""
