from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from ..capsule.registry import index_path
from ..core.registry import Registry
from ..core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    plugin_files_fingerprint,
    reset_capsule_plugin_cache,
)


OPTIMIZER_REGISTRY = Registry("optimizers", package="lelabo.optimizers")
_BASE_OPTIMIZER_ITEMS: dict[str, Any] | None = None
_LAST_OPTIMIZER_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_OPTIMIZER_ITEMS: dict[str, Any] | None = None


@dataclass
class OptimizerContext:
    params: Iterable
    lr: float
    weight_decay: float = 0.0
    momentum: float = 0.9
    args: Any | None = None
    mode: str | None = None
    dataset: str | None = None
    params_extra: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None

    def optimizer_params(self) -> dict[str, Any]:
        return dict(self.params_extra or {})


def register_optimizer(name: str):
    return OPTIMIZER_REGISTRY.register(name)


def _ensure_optimizer_baseline() -> None:
    global _BASE_OPTIMIZER_ITEMS
    if _BASE_OPTIMIZER_ITEMS is not None:
        return
    _BASE_OPTIMIZER_ITEMS = OPTIMIZER_REGISTRY.snapshot_discovered_items()


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
        plugin_files_fingerprint(active_root, kinds=("optimizers",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("optimizers",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_optimizer_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_OPTIMIZER_REFRESH_KEY, _LAST_OPTIMIZER_ITEMS
    _ensure_optimizer_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_OPTIMIZER_REFRESH_KEY == refresh_key and _LAST_OPTIMIZER_ITEMS is not None:
        OPTIMIZER_REGISTRY._items = dict(_LAST_OPTIMIZER_ITEMS)
        return

    OPTIMIZER_REGISTRY._items = dict(_BASE_OPTIMIZER_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("optimizers",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("optimizers",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("optimizers",), capsules_dir=capsules_dir)
    _LAST_OPTIMIZER_REFRESH_KEY = refresh_key
    _LAST_OPTIMIZER_ITEMS = dict(OPTIMIZER_REGISTRY._items)


def build_optimizer(name: str, ctx: OptimizerContext) -> torch.optim.Optimizer:
    _refresh_optimizer_registry()
    builder = OPTIMIZER_REGISTRY.get(name)
    out = builder(ctx)
    if not isinstance(out, torch.optim.Optimizer):
        raise TypeError(f"Optimizer builder '{name}' must return torch.optim.Optimizer, got {type(out).__name__}.")
    return out


def get_optimizer_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    _refresh_optimizer_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return OPTIMIZER_REGISTRY.names()


def make_optimizer(
    name: str,
    params: Iterable,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    *,
    args: Any | None = None,
    mode: str | None = None,
    dataset: str | None = None,
    extra: dict[str, Any] | None = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    raw_name = str(name).strip().lower()
    if not raw_name:
        raise ValueError("Optimizer name cannot be empty.")
    ctx = OptimizerContext(
        params=params,
        lr=float(lr),
        weight_decay=float(weight_decay),
        momentum=float(momentum),
        args=args,
        mode=mode,
        dataset=dataset,
        params_extra=dict(kwargs),
        extra=dict(extra or {}),
    )
    return build_optimizer(raw_name, ctx)

