from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .controller import SchedulerController


SCHEDULER_REGISTRY = Registry("schedulers", package="lab.schedulers")
_BASE_SCHEDULER_ITEMS: dict[str, Any] | None = None
_LAST_SCHEDULER_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_SCHEDULER_ITEMS: dict[str, Any] | None = None


@dataclass
class SchedulerContext:
    optimizer: torch.optim.Optimizer
    args: Any | None = None
    epochs: int | None = None
    steps_per_epoch: int | None = None
    interval: str = "epoch"
    monitor: str = "val.loss"
    params: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None

    @property
    def total_steps(self) -> int | None:
        if self.epochs is None or self.steps_per_epoch is None:
            return None
        try:
            e = int(self.epochs)
            s = int(self.steps_per_epoch)
            if e <= 0 or s <= 0:
                return None
            return int(e * s)
        except Exception:
            return None

    def scheduler_params(self) -> dict[str, Any]:
        return dict(self.params or {})


def register_scheduler(name: str):
    return SCHEDULER_REGISTRY.register(name)


def _ensure_scheduler_baseline() -> None:
    global _BASE_SCHEDULER_ITEMS
    if _BASE_SCHEDULER_ITEMS is not None:
        return
    _BASE_SCHEDULER_ITEMS = SCHEDULER_REGISTRY.snapshot_discovered_items()


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
        plugin_files_fingerprint(active_root, kinds=("schedulers",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("schedulers",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_scheduler_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_SCHEDULER_REFRESH_KEY, _LAST_SCHEDULER_ITEMS
    _ensure_scheduler_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_SCHEDULER_REFRESH_KEY == refresh_key and _LAST_SCHEDULER_ITEMS is not None:
        SCHEDULER_REGISTRY._items = dict(_LAST_SCHEDULER_ITEMS)
        return

    SCHEDULER_REGISTRY._items = dict(_BASE_SCHEDULER_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("schedulers",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("schedulers",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("schedulers",), capsules_dir=capsules_dir)
    _LAST_SCHEDULER_REFRESH_KEY = refresh_key
    _LAST_SCHEDULER_ITEMS = dict(SCHEDULER_REGISTRY._items)


def build_scheduler(name: str, ctx: SchedulerContext):
    _refresh_scheduler_registry()
    builder = SCHEDULER_REGISTRY.get(name)
    out = builder(ctx)
    if out is None:
        return None
    if isinstance(out, SchedulerController):
        return out
    return SchedulerController(
        out,
        interval=ctx.interval,
        monitor=ctx.monitor,
    )


def get_scheduler_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    _refresh_scheduler_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return SCHEDULER_REGISTRY.names()


def make_scheduler(
    name: str | None,
    optimizer: torch.optim.Optimizer,
    *,
    args: Any | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    interval: str = "epoch",
    monitor: str = "val.loss",
    **kwargs: Any,
):
    if name is None:
        return None
    raw_name = str(name).strip().lower()
    if raw_name in {"", "none", "null", "off"}:
        return None

    ctx = SchedulerContext(
        optimizer=optimizer,
        args=args,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        interval=str(interval),
        monitor=str(monitor),
        params=dict(kwargs),
    )
    return build_scheduler(raw_name, ctx)

