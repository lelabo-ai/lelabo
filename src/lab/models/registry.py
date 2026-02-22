from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from lab.core.registry import Registry
from ..core.utils.capsule_plugins import (
    load_capsule_plugins,
    load_installed_capsule_plugins,
    reset_capsule_plugin_cache,
)

MODEL_REGISTRY = Registry("models", package="lab.models.imported")
_BASE_MODEL_ITEMS: dict[str, Any] | None = None

@dataclass
class ModelContext:
    dataset: str
    num_classes: int
    in_dim: int | None = None
    in_channels: int | None = None
    input_shape: tuple[int, ...] | None = None
    extra: dict[str, Any] | None = None


def register_model(name: str):
    return MODEL_REGISTRY.register(name)


def _ensure_model_baseline() -> None:
    global _BASE_MODEL_ITEMS
    if _BASE_MODEL_ITEMS is not None:
        return
    _BASE_MODEL_ITEMS = MODEL_REGISTRY.snapshot_discovered_items()


def _refresh_model_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    _ensure_model_baseline()
    MODEL_REGISTRY._items = dict(_BASE_MODEL_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("models",))
    for root in list(extra_capsule_roots or []):
        load_capsule_plugins(kinds=("models",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("models",), capsules_dir=capsules_dir)


def build_model(name: str, ctx: ModelContext, args):
    _refresh_model_registry()
    builder = MODEL_REGISTRY.get(name)
    return builder(ctx, args)


def get_model_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
):
    _refresh_model_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return MODEL_REGISTRY.names()
