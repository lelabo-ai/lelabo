from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ...capsule.registry import get_capsule
from ...core.utils.capsule_plugins import find_active_capsule_root, load_capsule_plugins
from ...metrics.registry import get_metric_names
from ...models.registry import get_model_names
from ...supervised.datasets.registry import get_dataset_names
from ...update_rules.registry import get_update_rule_names


LIST_HELP = """\
List available LeLabo registries (algos, datasets, models, metrics).

Usage:
  lelabo list [target] [--json]
  lelabo ls [target] [--json]

Targets:
  all        Show all registries (default)
  algos      Show update-rule algorithm names
  datasets   Show dataset names
  models     Show model names
  metrics    Show metric names

Examples:
  lelabo list
  lelabo list algos
  lelabo list datasets --json
  lelabo list models --capsule my_capsule_alias

Notes:
  - `lelabo list` automatically includes built-ins + installed capsules.
  - Use `--capsule` to additionally include local/uninstalled capsules.
"""


_TARGET_ALIASES = {
    "all": "all",
    "algo": "algos",
    "algos": "algos",
    "rule": "algos",
    "rules": "algos",
    "update-rule": "algos",
    "update-rules": "algos",
    "dataset": "datasets",
    "datasets": "datasets",
    "model": "models",
    "models": "models",
    "metric": "metrics",
    "metrics": "metrics",
}


def _normalized_target(raw: str) -> str:
    key = str(raw).strip().lower()
    target = _TARGET_ALIASES.get(key)
    if target is None:
        raise ValueError(f"Unknown list target '{raw}'. Use one of: all, algos, datasets, models, metrics.")
    return target


def _collect(target: str, *, capsules_dir: Path | None) -> dict[str, list[str]]:
    if target == "algos":
        return {"algos": sorted(get_update_rule_names(capsules_dir=capsules_dir))}
    if target == "datasets":
        return {"datasets": sorted(get_dataset_names(capsules_dir=capsules_dir))}
    if target == "models":
        return {"models": sorted(get_model_names(capsules_dir=capsules_dir))}
    if target == "metrics":
        return {"metrics": sorted(get_metric_names(capsules_dir=capsules_dir))}
    return {
        "algos": sorted(get_update_rule_names(capsules_dir=capsules_dir)),
        "datasets": sorted(get_dataset_names(capsules_dir=capsules_dir)),
        "models": sorted(get_model_names(capsules_dir=capsules_dir)),
        "metrics": sorted(get_metric_names(capsules_dir=capsules_dir)),
    }


def _kinds_for_target(target: str) -> tuple[str, ...]:
    if target == "algos":
        return ("update_rules",)
    if target == "datasets":
        return ("datasets",)
    if target == "models":
        return ("models",)
    if target == "metrics":
        return ("metrics",)
    return ("models", "update_rules", "datasets", "metrics")


def _resolve_capsule_root(ref: str, capsules_dir: Path | None) -> Path:
    as_path = Path(ref).expanduser()
    if as_path.exists():
        start = as_path.resolve()
        if start.is_file():
            start = start.parent
        root = find_active_capsule_root(start=start)
        if root is None:
            raise ValueError(f"Path '{ref}' is not inside a capsule (missing capsule.toml).")
        return root

    row = get_capsule(ref, capsules_dir)
    if row is None:
        raise ValueError(f"Unknown capsule '{ref}' (not found as path nor installed id/alias).")
    root = Path(str(row.get("path", ""))).resolve()
    if not root.exists():
        raise ValueError(f"Capsule path does not exist on disk: {root}")
    return root


def _load_explicit_capsules(
    refs: Sequence[str],
    *,
    target: str,
    capsules_dir: Path | None,
) -> None:
    if not refs:
        return
    kinds = _kinds_for_target(target)
    for ref in refs:
        root = _resolve_capsule_root(ref, capsules_dir)
        load_capsule_plugins(kinds=kinds, capsule_root=root)


def _print_text(rows: dict[str, list[str]]) -> None:
    order = ("algos", "datasets", "models", "metrics")
    first = True
    for key in order:
        if key not in rows:
            continue
        values = rows[key]
        if not first:
            print("")
        first = False
        print(f"{key} ({len(values)})")
        for item in values:
            print(f"- {item}")
        if not values:
            print("- (none)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lelabo list")
    parser.add_argument("target", nargs="?", default="all", help="all|algos|datasets|models|metrics")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--capsule",
        action="append",
        default=[],
        metavar="PATH_OR_ID",
        help="Load plugins from a capsule path or an installed capsule id/alias (repeatable)",
    )
    parser.add_argument(
        "--capsules-dir",
        default=None,
        help="Capsule store path used to resolve `--capsule <id_or_alias>`",
    )
    return parser


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if args and args[0] in {"-h", "--help", "help"}:
        print(LIST_HELP)
        return 0

    parsed = _build_parser().parse_args(args)
    try:
        target = _normalized_target(parsed.target)
    except ValueError as exc:
        raise SystemExit(str(exc))

    caps_dir = Path(parsed.capsules_dir) if parsed.capsules_dir else None
    try:
        _load_explicit_capsules(
            [str(x) for x in parsed.capsule],
            target=target,
            capsules_dir=caps_dir,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    rows = _collect(target, capsules_dir=caps_dir)
    if bool(parsed.json):
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        _print_text(rows)
    return 0


__all__ = ["main"]
