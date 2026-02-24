"""Public update-rule API."""

from __future__ import annotations

from .registry import UpdateRuleContext, build_update_rule, get_update_rule_names, register_update_rule
from . import builders  # noqa: F401

__all__ = [
    "UpdateRuleContext",
    "register_update_rule",
    "build_update_rule",
    "get_update_rule_names",
]
