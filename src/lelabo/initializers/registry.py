"""Registry and context helpers for parameter initializers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.registry import Registry
from ..capsule.plugins.snapshot import build_capsule_registry_snapshot


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

def _initializer_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_initializer_baseline()
    return build_capsule_registry_snapshot(
        registry_name="initializers",
        kind="initializers",
        builtins=dict(_BASE_INITIALIZER_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def build_initializer(name: str, ctx: InitializerContext) -> Callable[[Any], Any]:
    builder = _initializer_snapshot().get(name)
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
    return _initializer_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()


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
