"""Grid sweep utilities: cartesian product, CLI arg generation, run naming."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any


def flagify(key: str) -> str:
    """Convert a YAML key like 'weight_decay' to a CLI flag '--weight-decay'."""
    return "--" + key.replace("_", "-")


def to_cli_args(d: dict[str, Any]) -> list[str]:
    """Convert a dict of key-value pairs to a flat list of CLI arguments."""
    args: list[str] = []
    for k, v in d.items():
        if v is None:
            continue
        f = flagify(k)
        if isinstance(v, bool):
            if v:
                args.append(f)
            continue
        args += [f, str(v)]
    return args


def cartesian_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations from a grid of parameter lists."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, prod)) for prod in itertools.product(*vals)]


def sanitize(val: Any) -> str:
    """Make a value filesystem-friendly for use in directory names."""
    s = str(val)
    for old, new in [
        ("/", "-"), ("\\", "-"), (" ", ""), (":", "-"),
        ("(", ""), (")", ""), ("[", ""), ("]", ""),
        ("{", ""), ("}", ""), (",", "_"),
    ]:
        s = s.replace(old, new)
    return s


def stable_short_id(args_dict: dict[str, Any], n: int = 6) -> str:
    """Deterministic short hash from a full args dict. Same args -> same id."""
    blob = json.dumps(args_dict, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:n]


def build_run_dirname(args_dict: dict[str, Any], display_keys: list[str]) -> str:
    """Human-readable run folder name with stable short id suffix."""
    parts: list[str] = []
    for k in display_keys:
        if k not in args_dict:
            continue
        parts.append(f"{k}={sanitize(args_dict[k])}")
    parts.append(f"id={stable_short_id(args_dict)}")
    return "__".join(parts)
