"""Model registry and builder context definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from lelabo.core.registry import Registry
from ..capsule.plugins.snapshot import build_capsule_registry_snapshot

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

def _model_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_model_baseline()
    return build_capsule_registry_snapshot(
        registry_name="models",
        kind="models",
        builtins=dict(_BASE_MODEL_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def build_model(name: str, ctx: ModelContext, args):
    builder = _model_snapshot().get(name)
    return builder(ctx, args)


def get_model_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
):
    return _model_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()
