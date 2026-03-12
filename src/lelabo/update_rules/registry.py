from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..core.registry import Registry
from ..capsule.plugins.snapshot import build_capsule_registry_snapshot


UPDATE_RULE_REGISTRY = Registry("update_rules", package="lelabo.update_rules.builtins")
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

def _update_rule_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_update_rule_baseline()
    return build_capsule_registry_snapshot(
        registry_name="update_rules",
        kind="update_rules",
        builtins=dict(_BASE_UPDATE_RULE_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def build_update_rule(name: str, ctx: UpdateRuleContext):
    builder = _update_rule_snapshot().get(name)
    return builder(ctx)


def get_update_rule_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
):
    return _update_rule_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()
