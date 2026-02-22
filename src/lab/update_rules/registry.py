from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.registry import Registry
from ..core.utils.capsule_plugins import (
    load_capsule_plugins,
    load_installed_capsule_plugins,
    reset_capsule_plugin_cache,
)


UPDATE_RULE_REGISTRY = Registry("update_rules", package="lab.update_rules")
_BASE_UPDATE_RULE_ITEMS: dict[str, Any] | None = None


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


def _refresh_update_rule_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    _ensure_update_rule_baseline()
    UPDATE_RULE_REGISTRY._items = dict(_BASE_UPDATE_RULE_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("update_rules",))
    for root in list(extra_capsule_roots or []):
        load_capsule_plugins(kinds=("update_rules",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("update_rules",), capsules_dir=capsules_dir)


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
