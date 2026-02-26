from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..capsule.registry import index_path
from ..core.registry import Registry
from ..core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    plugin_files_fingerprint,
    reset_capsule_plugin_cache,
)


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
        plugin_files_fingerprint(active_root, kinds=("callbacks",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("callbacks",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_callback_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_CALLBACK_REFRESH_KEY, _LAST_CALLBACK_ITEMS
    _ensure_callback_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_CALLBACK_REFRESH_KEY == refresh_key and _LAST_CALLBACK_ITEMS is not None:
        CALLBACK_REGISTRY._items = dict(_LAST_CALLBACK_ITEMS)
        return

    CALLBACK_REGISTRY._items = dict(_BASE_CALLBACK_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("callbacks",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("callbacks",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("callbacks",), capsules_dir=capsules_dir)
    _LAST_CALLBACK_REFRESH_KEY = refresh_key
    _LAST_CALLBACK_ITEMS = dict(CALLBACK_REGISTRY._items)


def build_callback(name: str, ctx: CallbackContext):
    _refresh_callback_registry()
    builder = CALLBACK_REGISTRY.get(name)
    return builder(ctx)


def get_callback_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    _refresh_callback_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return CALLBACK_REGISTRY.names()

