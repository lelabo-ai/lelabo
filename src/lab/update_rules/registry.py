from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.registry import Registry
from ..core.utils.capsule_plugins import load_capsule_plugins, load_installed_capsule_plugins


UPDATE_RULE_REGISTRY = Registry("update_rules", package="lab.update_rules")


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

def _load_update_rule_capsule_plugins(*, capsules_dir: Path | None = None) -> None:
    load_capsule_plugins(kinds=("update_rules",))
    load_installed_capsule_plugins(kinds=("update_rules",), capsules_dir=capsules_dir)


def build_update_rule(name: str, ctx: UpdateRuleContext):
    if not UPDATE_RULE_REGISTRY.names():
        UPDATE_RULE_REGISTRY.discover()
    _load_update_rule_capsule_plugins()
    builder = UPDATE_RULE_REGISTRY.get(name)
    return builder(ctx)


def get_update_rule_names(*, capsules_dir: Path | None = None):
    if not UPDATE_RULE_REGISTRY.names():
        UPDATE_RULE_REGISTRY.discover()
    _load_update_rule_capsule_plugins(capsules_dir=capsules_dir)
    return UPDATE_RULE_REGISTRY.names()
