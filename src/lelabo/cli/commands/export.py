"""Local capsule bundle export command."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any, Sequence

from ...capsule import get_capsule
from ...capsule.discovery import discover_visible_capsules, find_capsule_root
from ...config.user_settings import load_effective_settings
from ..ui import print_block, print_status


EXPORT_JSON_SCHEMA = "lelabo.cli.export/v1"


EXPORT_HELP = """\
Export one capsule as a local `.tar.gz` bundle.

Usage:
  lelabo export <capsule> [--out PATH] [--json]

Notes:
  - pass a capsule id or alias from `lelabo capsule list`
  - local filesystem paths are not accepted by `lelabo export` in v1
"""


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _effective_settings() -> dict[str, Any]:
    return load_effective_settings()


def _resolved_capsules_dir(raw_capsules_dir: str | None, settings: dict[str, Any]) -> Path | None:
    if raw_capsules_dir:
        return Path(raw_capsules_dir).expanduser().resolve()
    cfg = settings.get("capsules", {})
    if not isinstance(cfg, dict):
        return None
    store_dir = str(cfg.get("store_dir", "") or "").strip()
    if not store_dir:
        return None
    return Path(store_dir).expanduser().resolve()


def _resolve_capsule_root(capsule_ref: str | None, *, caps_dir: Path | None) -> Path:
    token = str(capsule_ref or "").strip()
    if not token:
        raise SystemExit("Missing capsule. Use `lelabo export <capsule>`. Run `lelabo capsule list` to inspect available capsules.")
    if "/" in token or token.startswith("."):
        raise SystemExit(
            "Local paths are not accepted by `lelabo export` in v1. "
            "Pass a capsule id or alias from `lelabo capsule list`."
        )

    visible = [
        item
        for item in discover_visible_capsules(start=Path.cwd(), capsules_dir=caps_dir)
        if item.capsule_id == token or token in set(item.aliases)
    ]
    if len(visible) == 1:
        return visible[0].root
    if len(visible) > 1:
        matches = ", ".join(item.path for item in visible)
        raise SystemExit(f"Capsule '{token}' is ambiguous across: {matches}")

    row = get_capsule(token, caps_dir)
    if row is None:
        raise SystemExit(f"Unknown capsule '{token}'. Run `lelabo capsule list` to inspect available capsules.")
    root = Path(str(row.get("path", ""))).expanduser().resolve()
    if not root.exists() or not root.is_dir() or find_capsule_root(root) is None:
        raise SystemExit(f"Capsule path does not exist on disk: {root}")
    return root


def _export_capsule_local_bundle(capsule_root: Path, *, out_path: Path | None) -> Path:
    root = capsule_root.resolve()
    out = (Path.cwd() / f"{root.name}.tar.gz").resolve() if out_path is None else out_path.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, mode="w:gz") as tf:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if ".git" in rel.parts or "__pycache__" in rel.parts:
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            tf.add(path, arcname=f"{root.name}/{rel.as_posix()}")
    return out


def _build_parser(*, prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Export one capsule as a local tar.gz bundle.",
        epilog=f"Examples:\n  {prog} my_capsule\n  {prog} my_capsule --out ./my_capsule.tar.gz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("capsule_ref", help="Capsule id or alias")
    parser.add_argument("--out", default=None, help="Output bundle path (.tar.gz)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    return parser


def run_export_command(argv: Sequence[str], *, prog: str = "lelabo export") -> tuple[int, dict[str, Any]]:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(list(argv))
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir)
    out = _export_capsule_local_bundle(
        capsule_root,
        out_path=Path(args.out).expanduser() if args.out else None,
    )
    payload = {
        "schema_version": EXPORT_JSON_SCHEMA,
        "command": "export",
        "capsule": {
            "capsule_id": str(get_capsule(str(args.capsule_ref), caps_dir).get("capsule_id", capsule_root.name)),
            "path": str(capsule_root),
        },
        "result": {
            "bundle_path": str(out),
            "format": "tar.gz",
        },
    }
    if bool(args.json):
        _print_json(payload)
    else:
        print_status("success", "Capsule exported locally.")
        print_block(
            "Export",
            (
                ("capsule", payload["capsule"]["capsule_id"]),
                ("bundle_path", out),
            ),
        )
    return 0, payload


def main(argv: Sequence[str]) -> int:
    if argv and str(argv[0]).strip() in {"-h", "--help", "help"}:
        print(EXPORT_HELP)
        return 0
    rc, _ = run_export_command(argv)
    return int(rc)


__all__ = [
    "main",
    "run_export_command",
]
