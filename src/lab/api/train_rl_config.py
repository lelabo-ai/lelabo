from __future__ import annotations

from typing import Any

from ..algorithms.rl.registry import get_rl_config_contract, resolve_rl_config


def parse_rl_param_overrides(raw_params: list[str] | tuple[str, ...]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in raw_params:
        entry = str(raw).strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid --rl-param '{entry}'. Expected KEY=VALUE.")
        key_raw, value_raw = entry.split("=", 1)
        key = key_raw.strip().replace("-", "_")
        value = value_raw.strip()
        if not key:
            raise ValueError(f"Invalid --rl-param '{entry}'. Empty key.")
        overrides[key] = value
    return overrides


def get_rl_algo_contract(algo: str) -> tuple[str, ...]:
    return get_rl_config_contract(algo)


def build_rl_algo_config(algo: str, overrides: dict[str, str]) -> Any:
    return resolve_rl_config(algo, overrides)


def build_dqn_config(overrides: dict[str, str]) -> Any:
    return build_rl_algo_config("dqn", overrides)


def build_ppo_config(overrides: dict[str, str]) -> Any:
    return build_rl_algo_config("ppo", overrides)

