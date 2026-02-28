"""Public loss factory API."""

from __future__ import annotations

from .registry import (
    LossContext,
    build_loss,
    get_loss_names,
    make_loss,
    register_loss,
)

# Force built-in loss registrations.
from . import builtins  # noqa: F401

__all__ = [
    "LossContext",
    "register_loss",
    "build_loss",
    "get_loss_names",
    "make_loss",
]
