"""LeLabo runtime primitives."""

from __future__ import annotations

import importlib

__all__ = [
    "accumulators",
    "activations",
    "batch",
    "callbacks",
    "display",
    "logger",
    "registry",
    "reporters",
    "seed",
    "steps",
    "state",
    "train_types",
    "trainer",
]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
