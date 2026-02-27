"""Public optimizer/scheduler factory API."""

from __future__ import annotations

from .factory import make_optimizer, make_scheduler
from .registry import (
    OptimizerContext,
    build_optimizer,
    get_optimizer_names,
    register_optimizer,
)

# Force built-in optimizer registrations.
from . import builtins  # noqa: F401

__all__ = [
    "OptimizerContext",
    "register_optimizer",
    "build_optimizer",
    "get_optimizer_names",
    "make_optimizer",
    "make_scheduler",
]
