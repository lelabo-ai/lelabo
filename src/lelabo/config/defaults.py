from __future__ import annotations

import copy
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

from .versioning import (
    TRAIN_CONFIG_SCHEMA_VERSION,
    resolve_config_version,
    resolve_lelabo_version,
)

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


def _normalize_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k).strip().replace("-", "_"): _normalize_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keys(v) for v in value]
    return value


@lru_cache(maxsize=1)
def _load_default_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    pkg = resources.files("lelabo.config.assets")
    supervised_path = pkg.joinpath("supervised.default.toml")
    rl_path = pkg.joinpath("rl.default.toml")

    with supervised_path.open("rb") as f:
        supervised_raw = tomllib.load(f)
    with rl_path.open("rb") as f:
        rl_raw = tomllib.load(f)

    supervised = _normalize_keys(dict(supervised_raw))
    rl = _normalize_keys(dict(rl_raw))
    supervised["config_version"] = resolve_config_version(
        supervised.get("config_version", TRAIN_CONFIG_SCHEMA_VERSION)
    )
    rl["config_version"] = resolve_config_version(
        rl.get("config_version", TRAIN_CONFIG_SCHEMA_VERSION)
    )
    supervised["lelabo_version"] = resolve_lelabo_version(supervised.get("lelabo_version", "auto"))
    rl["lelabo_version"] = resolve_lelabo_version(rl.get("lelabo_version", "auto"))
    supervised.setdefault("metrics", [])
    return supervised, rl


def get_default_supervised_config() -> dict[str, Any]:
    cfg, _ = _load_default_bundle()
    return copy.deepcopy(cfg)


def get_default_rl_config() -> dict[str, Any]:
    _, cfg = _load_default_bundle()
    return copy.deepcopy(cfg)


DEFAULT_SUPERVISED_CONFIG = get_default_supervised_config()
DEFAULT_RL_CONFIG = get_default_rl_config()
