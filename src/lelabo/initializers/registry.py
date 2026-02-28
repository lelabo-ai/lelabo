from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from ..capsule.registry import index_path
from ..core.registry import Registry
from ..core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    plugin_files_fingerprint,
    reset_capsule_plugin_cache,
)


INITIALIZER_REGISTRY = Registry("initializers", package="lelabo.initializers")
_BASE_INITIALIZER_ITEMS: dict[str, Any] | None = None
_LAST_INITIALIZER_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_INITIALIZER_ITEMS: dict[str, Any] | None = None


@dataclass
class InitializerContext:
    args: Any | None = None
    mode: str | None = None
    dataset: str | None = None
    model_name: str | None = None
    params: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None

    def initializer_params(self) -> dict[str, Any]:
        return dict(self.params or {})


def register_initializer(name: str):
    return INITIALIZER_REGISTRY.register(name)


def _ensure_initializer_baseline() -> None:
    global _BASE_INITIALIZER_ITEMS
    if _BASE_INITIALIZER_ITEMS is not None:
        return
    _BASE_INITIALIZER_ITEMS = INITIALIZER_REGISTRY.snapshot_discovered_items()


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
        plugin_files_fingerprint(active_root, kinds=("initializers",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("initializers",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_initializer_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_INITIALIZER_REFRESH_KEY, _LAST_INITIALIZER_ITEMS
    _ensure_initializer_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_INITIALIZER_REFRESH_KEY == refresh_key and _LAST_INITIALIZER_ITEMS is not None:
        INITIALIZER_REGISTRY._items = dict(_LAST_INITIALIZER_ITEMS)
        return

    INITIALIZER_REGISTRY._items = dict(_BASE_INITIALIZER_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("initializers",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("initializers",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("initializers",), capsules_dir=capsules_dir)
    _LAST_INITIALIZER_REFRESH_KEY = refresh_key
    _LAST_INITIALIZER_ITEMS = dict(INITIALIZER_REGISTRY._items)


def build_initializer(name: str, ctx: InitializerContext) -> Callable[[Any], Any]:
    _refresh_initializer_registry()
    builder = INITIALIZER_REGISTRY.get(name)
    out = builder(ctx)
    if not callable(out):
        raise TypeError(
            f"Initializer builder '{name}' must return a callable, got {type(out).__name__}."
        )
    return out


def get_initializer_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    _refresh_initializer_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return INITIALIZER_REGISTRY.names()


def make_initializer(
    name: str,
    *,
    args: Any | None = None,
    mode: str | None = None,
    dataset: str | None = None,
    model_name: str | None = None,
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
):
    raw_name = str(name).strip().lower()
    if not raw_name:
        raise ValueError("Initializer name cannot be empty.")
    ctx = InitializerContext(
        args=args,
        mode=mode,
        dataset=dataset,
        model_name=model_name,
        params=dict(params or {}),
        extra=dict(extra or {}),
    )
    return build_initializer(raw_name, ctx)
