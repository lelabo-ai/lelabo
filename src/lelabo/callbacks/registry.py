from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.registry import Registry
from ..core.utils.capsule_registry_snapshot import build_capsule_registry_snapshot


CALLBACK_REGISTRY = Registry("callbacks", package="lelabo.callbacks")
_BASE_CALLBACK_ITEMS: dict[str, Any] | None = None
_LAST_CALLBACK_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_CALLBACK_ITEMS: dict[str, Any] | None = None


@dataclass
class CallbackContext:
    args: Any
    mode: str = "supervised"
    dataset: str | None = None
    model: Any | None = None
    optimizer: Any | None = None
    schedulers: list[Any] | None = None
    params: dict[str, Any] | None = None
    index: int | None = None
    extra: dict[str, Any] | None = None

    def callback_params(self) -> dict[str, Any]:
        return dict(self.params or {})

    def callback_param(self, key: str, default: Any = None) -> Any:
        params = self.callback_params()
        return params.get(str(key), default)

    def namespaced_params(self, *aliases: str) -> dict[str, Any]:
        extra = self.extra if isinstance(self.extra, dict) else {}
        all_params = extra.get("callback_params", {})
        if not isinstance(all_params, Mapping):
            return {}
        for name in aliases:
            node = all_params.get(str(name))
            if isinstance(node, Mapping):
                return dict(node)
        return {}


def register_callback(name: str):
    return CALLBACK_REGISTRY.register(name)


def _ensure_callback_baseline() -> None:
    global _BASE_CALLBACK_ITEMS
    if _BASE_CALLBACK_ITEMS is not None:
        return
    _BASE_CALLBACK_ITEMS = CALLBACK_REGISTRY.snapshot_discovered_items()

def _callback_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_callback_baseline()
    return build_capsule_registry_snapshot(
        registry_name="callbacks",
        kind="callbacks",
        builtins=dict(_BASE_CALLBACK_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def build_callback(name: str, ctx: CallbackContext):
    builder = _callback_snapshot().get(name)
    return builder(ctx)


def get_callback_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    return _callback_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()
