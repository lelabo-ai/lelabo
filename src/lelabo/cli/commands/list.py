from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ...capsule.registry import get_capsule
from ...core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    reset_capsule_plugin_cache,
)
from ...metrics.registry import get_metric_names
from ...models.registry import get_model_names
from ...optimizers.registry import get_optimizer_names
from ...schedulers.registry import get_scheduler_names
from ...supervised.datasets.registry import get_dataset_names
from ...update_rules.registry import get_update_rule_names
from ...metrics import registry as metrics_registry
from ...models import registry as models_registry
from ...optimizers import registry as optimizers_registry
from ...schedulers import registry as schedulers_registry
from ...supervised.datasets import registry as datasets_registry
from ...update_rules import registry as update_rules_registry


LIST_HELP = """\
List available LeLabo registries (update_rules, datasets, models, optimizers, metrics, schedulers).

Usage:
  lelabo list [target] [--json]
  lelabo ls [target] [--json]

Targets:
  all        Show all registries (default)
  update-rules  Show update-rule names
  datasets   Show dataset names
  models     Show model names
  optimizers Show optimizer names
  metrics    Show metric names
  schedulers Show scheduler names

Examples:
  lelabo list
  lelabo list update-rules
  lelabo list datasets --json
  lelabo list models --capsule my_capsule_alias

Notes:
  - `lelabo list` automatically includes built-ins + installed capsules.
  - Use `--capsule` to additionally include local/uninstalled capsules.
"""


_TARGET_ALIASES = {
    "all": "all",
    "algo": "update_rules",
    "algos": "update_rules",
    "rule": "update_rules",
    "rules": "update_rules",
    "update-rule": "update_rules",
    "update-rules": "update_rules",
    "update_rules": "update_rules",
    "dataset": "datasets",
    "datasets": "datasets",
    "model": "models",
    "models": "models",
    "optimizer": "optimizers",
    "optimizers": "optimizers",
    "metric": "metrics",
    "metrics": "metrics",
    "scheduler": "schedulers",
    "schedulers": "schedulers",
}


def _normalized_target(raw: str) -> str:
    key = str(raw).strip().lower()
    target = _TARGET_ALIASES.get(key)
    if target is None:
        raise ValueError(
            f"Unknown list target '{raw}'. Use one of: all, update-rules, datasets, models, optimizers, metrics, schedulers."
        )
    return target


def _collect(
    target: str,
    *,
    capsules_dir: Path | None,
    explicit_capsule_roots: Sequence[Path] | None = None,
) -> dict[str, object]:
    extra_roots = list(explicit_capsule_roots or [])
    if target == "all":
        return _collect_all_single_refresh(
            capsules_dir=capsules_dir,
            explicit_capsule_roots=extra_roots,
        )
    if target == "update_rules":
        return {
            "update_rules": sorted(
                get_update_rule_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)
            )
        }
    if target == "datasets":
        return {"datasets": sorted(get_dataset_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots))}
    if target == "models":
        return {"models": sorted(get_model_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots))}
    if target == "optimizers":
        return {
            "optimizers": sorted(
                get_optimizer_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)
            )
        }
    if target == "metrics":
        return {"metrics": sorted(get_metric_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots))}
    if target == "schedulers":
        return {
            "schedulers": sorted(
                get_scheduler_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)
            )
        }
    return {
        "update_rules": sorted(
            get_update_rule_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)
        ),
        "datasets": sorted(get_dataset_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)),
        "models": sorted(get_model_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)),
        "optimizers": sorted(get_optimizer_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)),
        "metrics": sorted(get_metric_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)),
        "schedulers": sorted(get_scheduler_names(capsules_dir=capsules_dir, extra_capsule_roots=extra_roots)),
    }


def _normalize_capsule_roots(roots: Sequence[Path] | None) -> tuple[Path, ...]:
    return tuple(sorted({Path(root).resolve() for root in list(roots or [])}, key=str))


def _collect_all_single_refresh(
    *,
    capsules_dir: Path | None,
    explicit_capsule_roots: Sequence[Path] | None = None,
) -> dict[str, list[str]]:
    roots = _normalize_capsule_roots(explicit_capsule_roots)

    # Reset each registry to built-in baseline once, then load plugins once for all kinds.
    update_rules_registry._ensure_update_rule_baseline()
    datasets_registry._ensure_dataset_baseline()
    models_registry._ensure_model_baseline()
    optimizers_registry._ensure_optimizer_baseline()
    metrics_registry._ensure_metric_baseline()
    schedulers_registry._ensure_scheduler_baseline()

    update_rules_registry.UPDATE_RULE_REGISTRY._items = dict(update_rules_registry._BASE_UPDATE_RULE_ITEMS or {})
    datasets_registry.DATASET_REGISTRY._items = dict(datasets_registry._BASE_DATASET_ITEMS or {})
    models_registry.MODEL_REGISTRY._items = dict(models_registry._BASE_MODEL_ITEMS or {})
    optimizers_registry.OPTIMIZER_REGISTRY._items = dict(optimizers_registry._BASE_OPTIMIZER_ITEMS or {})
    metrics_registry.METRIC_REGISTRY._items = dict(metrics_registry._BASE_METRIC_ITEMS or {})
    schedulers_registry.SCHEDULER_REGISTRY._items = dict(schedulers_registry._BASE_SCHEDULER_ITEMS or {})

    reset_capsule_plugin_cache()
    kinds = ("update_rules", "datasets", "models", "optimizers", "metrics", "schedulers")
    load_capsule_plugins(kinds=kinds)
    for root in roots:
        load_capsule_plugins(kinds=kinds, capsule_root=root)
    load_installed_capsule_plugins(kinds=kinds, capsules_dir=capsules_dir)

    # Keep per-registry caches hot for subsequent calls in the same process.
    update_rules_registry._LAST_UPDATE_RULE_REFRESH_KEY = update_rules_registry._build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=roots,
    )
    update_rules_registry._LAST_UPDATE_RULE_ITEMS = dict(update_rules_registry.UPDATE_RULE_REGISTRY._items)

    datasets_registry._LAST_DATASET_REFRESH_KEY = datasets_registry._build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=roots,
    )
    datasets_registry._LAST_DATASET_ITEMS = dict(datasets_registry.DATASET_REGISTRY._items)

    models_registry._LAST_MODEL_REFRESH_KEY = models_registry._build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=roots,
    )
    models_registry._LAST_MODEL_ITEMS = dict(models_registry.MODEL_REGISTRY._items)
    optimizers_registry._LAST_OPTIMIZER_REFRESH_KEY = optimizers_registry._build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=roots,
    )
    optimizers_registry._LAST_OPTIMIZER_ITEMS = dict(optimizers_registry.OPTIMIZER_REGISTRY._items)

    metrics_registry._LAST_METRIC_REFRESH_KEY = metrics_registry._build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=roots,
    )
    metrics_registry._LAST_METRIC_ITEMS = dict(metrics_registry.METRIC_REGISTRY._items)
    schedulers_registry._LAST_SCHEDULER_REFRESH_KEY = schedulers_registry._build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=roots,
    )
    schedulers_registry._LAST_SCHEDULER_ITEMS = dict(schedulers_registry.SCHEDULER_REGISTRY._items)

    return {
        "update_rules": update_rules_registry.UPDATE_RULE_REGISTRY.names(),
        "datasets": datasets_registry.DATASET_REGISTRY.names(),
        "models": models_registry.MODEL_REGISTRY.names(),
        "optimizers": optimizers_registry.OPTIMIZER_REGISTRY.names(),
        "metrics": metrics_registry.METRIC_REGISTRY.names(),
        "schedulers": schedulers_registry.SCHEDULER_REGISTRY.names(),
    }


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


def _resolve_explicit_capsule_roots(
    refs: Sequence[str],
    *,
    capsules_dir: Path | None,
) -> list[Path]:
    if not refs:
        return []
    out: list[Path] = []
    for ref in refs:
        root = _resolve_capsule_root(ref, capsules_dir)
        out.append(root)
    return out


def _print_text(rows: dict[str, list[str]]) -> None:
    order = ("update_rules", "datasets", "models", "optimizers", "metrics", "schedulers")
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
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="all|update-rules|datasets|models|optimizers|metrics|schedulers",
    )
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
        explicit_roots = _resolve_explicit_capsule_roots(
            [str(x) for x in parsed.capsule],
            capsules_dir=caps_dir,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    rows = _collect(target, capsules_dir=caps_dir, explicit_capsule_roots=explicit_roots)
    if bool(parsed.json):
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        _print_text(rows)
    return 0


__all__ = ["main"]
