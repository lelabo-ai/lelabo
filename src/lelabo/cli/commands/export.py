"""Local capsule bundle export command."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any, Sequence

from ...capsule.discovery import resolve_visible_capsule_ref
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


def _resolve_visible_capsule(capsule_ref: str | None, *, caps_dir: Path | None) -> DiscoveredCapsule:
    try:
        return resolve_visible_capsule_ref(
            capsule_ref,
            start=Path.cwd(),
            capsules_dir=caps_dir,
            command="lelabo export",
            usage="lelabo export <capsule>",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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
    capsule = _resolve_visible_capsule(args.capsule_ref, caps_dir=caps_dir)
    capsule_root = capsule.root
    out = _export_capsule_local_bundle(
        capsule_root,
        out_path=Path(args.out).expanduser() if args.out else None,
    )
    payload = {
        "schema_version": EXPORT_JSON_SCHEMA,
        "command": "export",
        "capsule": {
            "capsule_id": capsule.capsule_id,
            "status": capsule.status,
            "aliases": list(capsule.aliases),
            "path": capsule.path,
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
                ("status", payload["capsule"]["status"]),
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
