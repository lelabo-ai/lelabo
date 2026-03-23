"""CLI entrypoint for listing built-in and capsule registry exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ...capsule import inspect_capsule_directory
from ...capsule.discovery import discover_workspace_capsules, find_capsule_root
from ...capsule.registry import get_capsule
from ...config.user_settings import load_effective_settings
from ...callbacks.registry import _callback_snapshot
from ...initializers.registry import _initializer_snapshot
from ...metrics.registry import _metric_snapshot
from ...losses.registry import _loss_snapshot
from ...models.registry import _model_snapshot
from ...optimizers.registry import _optimizer_snapshot
from ...schedulers.registry import _scheduler_snapshot
from ...supervised.datasets.registry import _dataset_snapshot
from ...update_rules.registry import _update_rule_snapshot
from ..ui import print_list_block


LIST_HELP = """\
List available LeLabo registries (update_rules, datasets, models, initializers, optimizers, losses, metrics, schedulers, callbacks).

Usage:
  lelabo list [target] [--json]
  lelabo ls [target] [--json]

Targets:
  all        Show all registries (default)
  update-rules  Show update-rule names
  datasets   Show dataset names
  models     Show model names
  initializers Show initializer names
  optimizers Show optimizer names
  losses     Show loss names
  metrics    Show metric names
  schedulers Show scheduler names
  callbacks  Show callback names

Examples:
  lelabo list
  lelabo list update-rules
  lelabo list datasets --json
  lelabo list models --capsule my_capsule_alias

Notes:
  - `lelabo list` includes built-ins plus capsules visible from the current workspace.
  - Stored capsules do not affect registries until checkout or explicit `--capsule ...`.
  - Use `--capsule` to additionally include a local capsule path or a stored capsule id/alias.
"""

LIST_JSON_SCHEMA = "lelabo.cli.list/v2"


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
    "initializer": "initializers",
    "initializers": "initializers",
    "optimizer": "optimizers",
    "optimizers": "optimizers",
    "loss": "losses",
    "losses": "losses",
    "metric": "metrics",
    "metrics": "metrics",
    "scheduler": "schedulers",
    "schedulers": "schedulers",
    "callback": "callbacks",
    "callbacks": "callbacks",
}

_REGISTRY_SPECS: dict[str, Any] = {
    "update_rules": _update_rule_snapshot,
    "datasets": _dataset_snapshot,
    "models": _model_snapshot,
    "initializers": _initializer_snapshot,
    "optimizers": _optimizer_snapshot,
    "losses": _loss_snapshot,
    "metrics": _metric_snapshot,
    "schedulers": _scheduler_snapshot,
    "callbacks": _callback_snapshot,
}


def _resolved_capsules_dir(raw_capsules_dir: str | None) -> Path | None:
    if raw_capsules_dir:
        return Path(raw_capsules_dir).expanduser().resolve()
    settings = load_effective_settings()
    cfg = settings.get("capsules", {})
    if not isinstance(cfg, dict):
        return None
    store_dir = str(cfg.get("store_dir", "") or "").strip()
    if not store_dir:
        return None
    return Path(store_dir).expanduser().resolve()


def _normalized_target(raw: str) -> str:
    key = str(raw).strip().lower()
    target = _TARGET_ALIASES.get(key)
    if target is None:
        raise ValueError(
            "Unknown list target "
            f"'{raw}'. Use one of: all, update-rules, datasets, models, initializers, optimizers, losses, metrics, schedulers, callbacks."
        )
    return target


def _resolve_capsule_root(ref: str, capsules_dir: Path | None) -> Path:
    as_path = Path(ref).expanduser()
    if as_path.exists():
        start = as_path.resolve()
        if start.is_file():
            start = start.parent
        root = find_capsule_root(start=start)
        if root is None:
            raise ValueError(f"Path '{ref}' is not inside a capsule (missing capsule.toml).")
        return root

    row = get_capsule(ref, capsules_dir)
    if row is None:
        raise ValueError(f"Unknown capsule '{ref}' (not found as path nor stored id/alias).")
    root = Path(str(row.get("path", ""))).resolve()
    if not root.exists():
        raise ValueError(f"Capsule path does not exist on disk: {root}")
    return root


def _resolve_explicit_capsule_roots(
    refs: Sequence[str],
    *,
    capsules_dir: Path | None,
) -> list[Path]:
    return [_resolve_capsule_root(ref, capsules_dir) for ref in refs]


def _capsule_name_for_root(root: str) -> str:
    path = Path(str(root)).expanduser().resolve()
    try:
        return str(inspect_capsule_directory(path).get("capsule_id", path.name)).strip() or path.name
    except Exception:
        return path.name


def _snapshot_rows(
    target: str,
    *,
    capsules_dir: Path | None,
    explicit_capsule_roots: Sequence[Path] | None = None,
) -> dict[str, dict[str, Any]]:
    extra_roots: list[Path] = []
    seen: set[str] = set()

    def _add_root(root: Path) -> None:
        token = str(root.resolve())
        if token in seen:
            return
        seen.add(token)
        extra_roots.append(root.resolve())

    for item in discover_workspace_capsules(start=Path.cwd()):
        _add_root(item.root)
    for root in list(explicit_capsule_roots or []):
        _add_root(root)

    def _grouped_row(name: str) -> dict[str, Any]:
        snapshot_builder = _REGISTRY_SPECS[name]
        snapshot = snapshot_builder(
            capsules_dir=capsules_dir,
            extra_capsule_roots=extra_roots,
        )
        capsule_rows: dict[str, list[str]] = {}
        for export in snapshot.capsule_exports.values():
            capsule_name = _capsule_name_for_root(export.capsule_root)
            capsule_rows.setdefault(capsule_name, []).append(str(export.name))
        return {
            "builtins": sorted(snapshot.builtins.keys()),
            "capsules": {
                key: sorted(set(values))
                for key, values in sorted(capsule_rows.items(), key=lambda item: item[0])
            },
        }

    if target == "all":
        return {name: _grouped_row(name) for name in _REGISTRY_SPECS}
    return {target: _grouped_row(target)}


def _print_grouped_text(rows: dict[str, dict[str, Any]]) -> None:
    order = (
        "update_rules",
        "datasets",
        "models",
        "initializers",
        "optimizers",
        "losses",
        "metrics",
        "schedulers",
        "callbacks",
    )
    first_section = True
    for key in order:
        if key not in rows:
            continue
        if not first_section:
            print("")
        first_section = False
        print(f"{key}:")
        grouped = rows[key]
        print_list_block("builtins:", list(grouped.get("builtins", []) or []))
        for capsule_name, items in dict(grouped.get("capsules", {}) or {}).items():
            print_list_block(f"{capsule_name}:", list(items or []))


def _list_json_payload(target: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if target == "all":
        return {
            "schema_version": LIST_JSON_SCHEMA,
            "target": "all",
            "registries": rows,
        }
    return {
        "schema_version": LIST_JSON_SCHEMA,
        "target": target,
        "sources": rows[target],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lelabo list")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="all|update-rules|datasets|models|initializers|optimizers|losses|metrics|schedulers|callbacks",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--capsule",
        action="append",
        default=[],
        metavar="PATH_OR_ID",
        help="Load plugins from a capsule path or a stored capsule id/alias (repeatable)",
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

    caps_dir = _resolved_capsules_dir(parsed.capsules_dir)
    try:
        explicit_roots = _resolve_explicit_capsule_roots(
            [str(x) for x in parsed.capsule],
            capsules_dir=caps_dir,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    rows = _snapshot_rows(target, capsules_dir=caps_dir, explicit_capsule_roots=explicit_roots)
    if bool(parsed.json):
        print(json.dumps(_list_json_payload(target, rows), indent=2, ensure_ascii=False))
    else:
        _print_grouped_text(rows)
    return 0


__all__ = ["main"]
