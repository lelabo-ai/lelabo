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


LOSS_REGISTRY = Registry("losses", package="lelabo.losses")
_BASE_LOSS_ITEMS: dict[str, Any] | None = None
_LAST_LOSS_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_LOSS_ITEMS: dict[str, Any] | None = None


@dataclass
class LossContext:
    args: Any | None = None
    mode: str | None = None
    dataset: str | None = None
    task: str | None = None
    num_classes: int | None = None
    params: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None

    def loss_params(self) -> dict[str, Any]:
        return dict(self.params or {})


def register_loss(name: str):
    return LOSS_REGISTRY.register(name)


def _ensure_loss_baseline() -> None:
    global _BASE_LOSS_ITEMS
    if _BASE_LOSS_ITEMS is not None:
        return
    _BASE_LOSS_ITEMS = LOSS_REGISTRY.snapshot_discovered_items()


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
        plugin_files_fingerprint(active_root, kinds=("losses",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("losses",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_loss_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_LOSS_REFRESH_KEY, _LAST_LOSS_ITEMS
    _ensure_loss_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_LOSS_REFRESH_KEY == refresh_key and _LAST_LOSS_ITEMS is not None:
        LOSS_REGISTRY._items = dict(_LAST_LOSS_ITEMS)
        return

    LOSS_REGISTRY._items = dict(_BASE_LOSS_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("losses",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("losses",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("losses",), capsules_dir=capsules_dir)
    _LAST_LOSS_REFRESH_KEY = refresh_key
    _LAST_LOSS_ITEMS = dict(LOSS_REGISTRY._items)


def build_loss(name: str, ctx: LossContext) -> Callable[[Any, Any], Any]:
    _refresh_loss_registry()
    builder = LOSS_REGISTRY.get(name)
    out = builder(ctx)
    if not callable(out):
        raise TypeError(f"Loss builder '{name}' must return a callable, got {type(out).__name__}.")
    return out


def get_loss_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    _refresh_loss_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return LOSS_REGISTRY.names()


def make_loss(
    name: str,
    *,
    args: Any | None = None,
    mode: str | None = None,
    dataset: str | None = None,
    task: str | None = None,
    num_classes: int | None = None,
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
):
    raw_name = str(name).strip().lower()
    if not raw_name:
        raise ValueError("Loss name cannot be empty.")
    ctx = LossContext(
        args=args,
        mode=mode,
        dataset=dataset,
        task=task,
        num_classes=num_classes,
        params=dict(params or {}),
        extra=dict(extra or {}),
    )
    return build_loss(raw_name, ctx)
