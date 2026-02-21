from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any


_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}


def _parse_bool(value: str, *, key: str) -> bool:
    text = str(value).strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    raise ValueError(
        f"Invalid boolean value for '{key}': '{value}'. "
        "Use one of: true/false, 1/0, yes/no, on/off."
    )


def _coerce_like_default(
    value: str,
    *,
    key: str,
    default: Any,
    allow_none_float: bool = False,
) -> Any:
    raw = str(value).strip()
    if default is None:
        if allow_none_float and raw.lower() in {"none", "null"}:
            return None
        if allow_none_float:
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"Invalid float value for '{key}': '{value}'.") from exc
        return raw

    if isinstance(default, bool):
        return _parse_bool(raw, key=key)
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid integer value for '{key}': '{value}'.") from exc
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid float value for '{key}': '{value}'.") from exc
    if isinstance(default, str):
        return raw
    raise ValueError(f"Unsupported override type for '{key}'.")


def _iter_leaf_paths(cfg: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    for field in fields(cfg):
        name = field.name
        value = getattr(cfg, name)
        path = f"{prefix}.{name}" if prefix else name
        if is_dataclass(value):
            out.extend(_iter_leaf_paths(value, prefix=path))
        else:
            out.append(path)
    return out


def list_contract_keys(cfg: Any, aliases: dict[str, str] | None = None) -> tuple[str, ...]:
    alias_map = dict(aliases or {})
    canonical = set(_iter_leaf_paths(cfg))
    return tuple(sorted(canonical | set(alias_map.keys())))


def resolve_dataclass_overrides(
    cfg: Any,
    overrides: dict[str, str],
    *,
    aliases: dict[str, str] | None = None,
    allow_none_float_paths: set[str] | None = None,
    algo_label: str,
) -> Any:
    alias_map = dict(aliases or {})
    none_float_paths = set(allow_none_float_paths or set())

    canonical_paths = set(_iter_leaf_paths(cfg))
    key_to_path = {path: path for path in canonical_paths}
    key_to_path.update(alias_map)
    available = sorted(set(key_to_path.keys()))

    for raw_key, value in overrides.items():
        key = str(raw_key).strip()
        path = key_to_path.get(key)
        if path is None:
            raise ValueError(
                f"Unsupported {algo_label} override '{raw_key}'. "
                f"Available keys: {available}"
            )

        obj = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)

        attr = parts[-1]
        default = getattr(obj, attr)
        allow_none_float = path in none_float_paths
        coerced = _coerce_like_default(
            value,
            key=path,
            default=default,
            allow_none_float=allow_none_float,
        )
        setattr(obj, attr, coerced)

    return cfg

