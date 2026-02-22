from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.registry import Registry


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


def build_update_rule(name: str, ctx: UpdateRuleContext):
    if not UPDATE_RULE_REGISTRY.names():
        UPDATE_RULE_REGISTRY.discover()
    builder = UPDATE_RULE_REGISTRY.get(name)
    return builder(ctx)


def get_update_rule_names():
    if not UPDATE_RULE_REGISTRY.names():
        UPDATE_RULE_REGISTRY.discover()
    return UPDATE_RULE_REGISTRY.names()
