from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

from lelabo.core.registry import Registry
from ..capsule.registry import index_path
from ..core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    plugin_files_fingerprint,
    reset_capsule_plugin_cache,
)

MODEL_REGISTRY = Registry("models", package="lelabo.models.builtins")
_BASE_MODEL_ITEMS: dict[str, Any] | None = None
_LAST_MODEL_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_MODEL_ITEMS: dict[str, Any] | None = None

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
        plugin_files_fingerprint(active_root, kinds=("models",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("models",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_model_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_MODEL_REFRESH_KEY, _LAST_MODEL_ITEMS
    _ensure_model_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_MODEL_REFRESH_KEY == refresh_key and _LAST_MODEL_ITEMS is not None:
        MODEL_REGISTRY._items = dict(_LAST_MODEL_ITEMS)
        return

    MODEL_REGISTRY._items = dict(_BASE_MODEL_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("models",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("models",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("models",), capsules_dir=capsules_dir)
    _LAST_MODEL_REFRESH_KEY = refresh_key
    _LAST_MODEL_ITEMS = dict(MODEL_REGISTRY._items)


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
