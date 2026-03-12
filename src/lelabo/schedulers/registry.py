from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..core.registry import Registry
from ..capsule.plugins.snapshot import build_capsule_registry_snapshot
from .controller import SchedulerController


SCHEDULER_REGISTRY = Registry("schedulers", package="lelabo.schedulers")
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

def _scheduler_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_scheduler_baseline()
    return build_capsule_registry_snapshot(
        registry_name="schedulers",
        kind="schedulers",
        builtins=dict(_BASE_SCHEDULER_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def build_scheduler(name: str, ctx: SchedulerContext):
    builder = _scheduler_snapshot().get(name)
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
    return _scheduler_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()


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
