"""Local capsule bundle export command."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Sequence

from ...capsule import get_capsule
from ...capsule.plugins.discovery import find_active_capsule_root
from ...config.user_settings import load_effective_settings
from ..interactive_picker import pick_many_with_checkboxes
from ..ui import print_block, print_status


EXPORT_JSON_SCHEMA = "lelabo.cli.export/v1"


EXPORT_HELP = """\
Export one capsule as a local `.tar.gz` bundle.

Usage:
  lelabo export [capsule_ref] [--out PATH] [--capsules-dir DIR] [--json]

Notes:
  - `capsule_ref` can be a local path or a stored capsule id/alias
  - if omitted, LeLabo uses the active capsule or a direct child capsule of the current workspace
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


def _is_capsule_root(path: Path) -> bool:
    return path.is_dir() and ((path / "capsule.toml").is_file() or (path / "manifest.json").is_file())


def _is_interactive_tty() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _workspace_child_capsules(root: Path) -> list[Path]:
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    return sorted(child.resolve() for child in children if child.is_dir() and _is_capsule_root(child))


def _pick_capsule_interactively(candidates: Sequence[Path]) -> Path:
    options = [(path.name, f"{path.name} | path: {path.name}") for path in candidates]
    selected = pick_many_with_checkboxes(
        title="Select capsule to export",
        text="Select one capsule to export as a tar.gz bundle.",
        options=options,
        selection_noun="capsule",
        confirm_button_text="Export selected",
        max_selection_count=1,
        max_selection_message="Select exactly one capsule to export.",
    )
    if selected is None:
        raise SystemExit("Export canceled by user.")
    chosen = str(selected[0]).strip() if selected else ""
    for path in candidates:
        if path.name == chosen:
            return path
    raise SystemExit("Export canceled by user.")


def _resolve_capsule_root(capsule_ref: str | None, *, caps_dir: Path | None) -> Path:
    if not capsule_ref:
        active = find_active_capsule_root()
        if active is not None:
            return active.resolve()
        direct_children = _workspace_child_capsules(Path.cwd())
        if len(direct_children) == 1:
            return direct_children[0]
        if not direct_children:
            raise SystemExit(
                "No capsule found in the current workspace. Pass a capsule path/id or create a capsule under this directory."
            )
        if _is_interactive_tty():
            return _pick_capsule_interactively(direct_children)
        names = ", ".join(path.name for path in direct_children)
        raise SystemExit(
            f"Multiple capsules found in the current workspace: {names}. Pass a capsule path/id to `lelabo export`."
        )

    ref_path = Path(capsule_ref).expanduser()
    if ref_path.exists():
        start = ref_path.resolve()
        if start.is_file():
            start = start.parent
        root = find_active_capsule_root(start=start)
        if root is None and _is_capsule_root(start):
            root = start
        if root is None:
            raise SystemExit(f"Path '{capsule_ref}' is not inside a capsule (missing capsule.toml).")
        return root.resolve()

    local_candidate = (Path.cwd() / str(capsule_ref)).resolve()
    if _is_capsule_root(local_candidate):
        return local_candidate

    row = get_capsule(capsule_ref, caps_dir)
    if row is None:
        raise SystemExit(f"Unknown capsule '{capsule_ref}' (not found as path nor stored id/alias).")
    root = Path(str(row.get("path", ""))).expanduser().resolve()
    if not root.exists() or not root.is_dir():
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
        epilog=f"Examples:\n  {prog}\n  {prog} my_capsule --out ./my_capsule.tar.gz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--out", default=None, help="Output bundle path (.tar.gz)")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    return parser


def run_export_command(argv: Sequence[str], *, prog: str = "lelabo export") -> tuple[int, dict[str, Any]]:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(list(argv))
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir)
    out = _export_capsule_local_bundle(
        capsule_root,
        out_path=Path(args.out).expanduser() if args.out else None,
    )
    payload = {
        "schema_version": EXPORT_JSON_SCHEMA,
        "command": "export",
        "capsule": {
            "capsule_id": capsule_root.name,
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
                ("capsule_path", capsule_root),
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
