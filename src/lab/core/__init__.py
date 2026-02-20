"""LeLabo core package.

Import submodules directly (for example ``lab.core.task``).
Submodules are loaded lazily to avoid pulling optional dependencies at import
time (notably RL dependencies such as ``gymnasium``).
"""

from __future__ import annotations

import importlib

__all__ = [
    "batch",
    "callbacks",
    "ppo_trainer",
    "replay_buffer",
    "rl_trainer",
    "robustness",
    "steps",
    "state",
    "task",
    "trainer",
    "utils",
]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
