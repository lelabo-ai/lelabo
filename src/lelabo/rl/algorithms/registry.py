"""Registry helpers for RL algorithms and their config contracts."""

from __future__ import annotations

from typing import Any, Callable

from .dqn import get_config_contract as get_dqn_config_contract
from .dqn import resolve_config_overrides as resolve_dqn_config_overrides
from .ppo import get_config_contract as get_ppo_config_contract
from .ppo import resolve_config_overrides as resolve_ppo_config_overrides


_CONFIG_RESOLVERS: dict[str, Callable[[dict[str, str]], Any]] = {
    "dqn": resolve_dqn_config_overrides,
    "ppo": resolve_ppo_config_overrides,
}

_CONFIG_CONTRACTS: dict[str, Callable[[], tuple[str, ...]]] = {
    "dqn": get_dqn_config_contract,
    "ppo": get_ppo_config_contract,
}


def get_rl_algo_names() -> tuple[str, ...]:
    return tuple(sorted(_CONFIG_RESOLVERS.keys()))


def get_rl_config_contract(algo: str) -> tuple[str, ...]:
    key = str(algo).strip().lower()
    fn = _CONFIG_CONTRACTS.get(key)
    if fn is None:
        raise ValueError(f"Unknown rl algo: {algo}")
    return fn()


def resolve_rl_config(algo: str, overrides: dict[str, str]) -> Any:
    key = str(algo).strip().lower()
    fn = _CONFIG_RESOLVERS.get(key)
    if fn is None:
        raise ValueError(f"Unknown rl algo: {algo}")
    return fn(overrides)
"""Registry helpers for RL algorithms and their config contracts."""
