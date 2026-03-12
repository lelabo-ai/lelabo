from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.registry import Registry
from ..capsule.plugins.snapshot import build_capsule_registry_snapshot


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

def _loss_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_loss_baseline()
    return build_capsule_registry_snapshot(
        registry_name="losses",
        kind="losses",
        builtins=dict(_BASE_LOSS_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def build_loss(name: str, ctx: LossContext) -> Callable[[Any, Any], Any]:
    builder = _loss_snapshot().get(name)
    out = builder(ctx)
    if not callable(out):
        raise TypeError(f"Loss builder '{name}' must return a callable, got {type(out).__name__}.")
    return out


def get_loss_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    return _loss_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()


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
