from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

from ..capsule.registry import index_path
from ..core.registry import Registry
from ..core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    plugin_files_fingerprint,
    reset_capsule_plugin_cache,
)


METRIC_REGISTRY = Registry("metrics", package="lab.metrics")
_BASE_METRIC_ITEMS: dict[str, Any] | None = None
_LAST_METRIC_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_METRIC_ITEMS: dict[str, Any] | None = None


@dataclass
class MetricContext:
    args: Any
    mode: str = "supervised"
    dataset: str | None = None
    algo: str | None = None
    extra: dict[str, Any] | None = None


def register_metric(name: str):
    return METRIC_REGISTRY.register(name)


def _ensure_metric_baseline() -> None:
    global _BASE_METRIC_ITEMS
    if _BASE_METRIC_ITEMS is not None:
        return
    _BASE_METRIC_ITEMS = METRIC_REGISTRY.snapshot_discovered_items()


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
        plugin_files_fingerprint(active_root, kinds=("metrics",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("metrics",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_metric_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_METRIC_REFRESH_KEY, _LAST_METRIC_ITEMS
    _ensure_metric_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_METRIC_REFRESH_KEY == refresh_key and _LAST_METRIC_ITEMS is not None:
        METRIC_REGISTRY._items = dict(_LAST_METRIC_ITEMS)
        return

    METRIC_REGISTRY._items = dict(_BASE_METRIC_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("metrics",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("metrics",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("metrics",), capsules_dir=capsules_dir)
    _LAST_METRIC_REFRESH_KEY = refresh_key
    _LAST_METRIC_ITEMS = dict(METRIC_REGISTRY._items)


def build_metric(name: str, ctx: MetricContext):
    _refresh_metric_registry()
    builder = METRIC_REGISTRY.get(name)
    return builder(ctx)


def get_metric_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    _refresh_metric_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return METRIC_REGISTRY.names()


def parse_metric_names(raw: Any) -> list[str]:
    if raw is None:
        return []

    chunks: list[str] = []
    if isinstance(raw, str):
        chunks = [raw]
    elif isinstance(raw, (list, tuple, set)):
        chunks = [str(item) for item in raw]
    else:
        chunks = [str(raw)]

    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for comma_part in str(chunk).split(","):
            for token in comma_part.strip().split():
                key = token.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(key)
    return out
