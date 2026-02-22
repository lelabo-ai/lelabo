from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

from ..core.registry import Registry
from ..capsule.registry import index_path
from ..core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    plugin_files_fingerprint,
    reset_capsule_plugin_cache,
)


UPDATE_RULE_REGISTRY = Registry("update_rules", package="lab.update_rules")
_BASE_UPDATE_RULE_ITEMS: dict[str, Any] | None = None
_LAST_UPDATE_RULE_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_UPDATE_RULE_ITEMS: dict[str, Any] | None = None


@dataclass
class UpdateRuleContext:
    args: Any
    model: Any
    task: Any | None
    optimizer: Any
    mode: str = "supervised"  # "supervised" | "rl" | custom
    dataset: str | None = None
    rl_algo: str | None = None
    extra: dict[str, Any] | None = None


def register_update_rule(name: str):
    return UPDATE_RULE_REGISTRY.register(name)


def _ensure_update_rule_baseline() -> None:
    global _BASE_UPDATE_RULE_ITEMS
    if _BASE_UPDATE_RULE_ITEMS is not None:
        return
    _BASE_UPDATE_RULE_ITEMS = UPDATE_RULE_REGISTRY.snapshot_discovered_items()


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
        plugin_files_fingerprint(active_root, kinds=("update_rules",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("update_rules",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_update_rule_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_UPDATE_RULE_REFRESH_KEY, _LAST_UPDATE_RULE_ITEMS
    _ensure_update_rule_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_UPDATE_RULE_REFRESH_KEY == refresh_key and _LAST_UPDATE_RULE_ITEMS is not None:
        UPDATE_RULE_REGISTRY._items = dict(_LAST_UPDATE_RULE_ITEMS)
        return

    UPDATE_RULE_REGISTRY._items = dict(_BASE_UPDATE_RULE_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("update_rules",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("update_rules",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("update_rules",), capsules_dir=capsules_dir)
    _LAST_UPDATE_RULE_REFRESH_KEY = refresh_key
    _LAST_UPDATE_RULE_ITEMS = dict(UPDATE_RULE_REGISTRY._items)


def build_update_rule(name: str, ctx: UpdateRuleContext):
    _refresh_update_rule_registry()
    builder = UPDATE_RULE_REGISTRY.get(name)
    return builder(ctx)


def get_update_rule_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
):
    _refresh_update_rule_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return UPDATE_RULE_REGISTRY.names()
