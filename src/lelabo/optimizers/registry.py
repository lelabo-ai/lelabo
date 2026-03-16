"""Optimizer registry and builder context helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from ..core.registry import Registry
from ..capsule.plugins.snapshot import build_capsule_registry_snapshot


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

def _optimizer_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_optimizer_baseline()
    return build_capsule_registry_snapshot(
        registry_name="optimizers",
        kind="optimizers",
        builtins=dict(_BASE_OPTIMIZER_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def build_optimizer(name: str, ctx: OptimizerContext) -> torch.optim.Optimizer:
    builder = _optimizer_snapshot().get(name)
    out = builder(ctx)
    if not isinstance(out, torch.optim.Optimizer):
        raise TypeError(f"Optimizer builder '{name}' must return torch.optim.Optimizer, got {type(out).__name__}.")
    return out


def get_optimizer_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    return _optimizer_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()


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
