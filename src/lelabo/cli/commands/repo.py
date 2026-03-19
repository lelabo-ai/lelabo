"""Advanced publish target management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ...capsule import get_capsule
from ...capsule.plugins.discovery import find_active_capsule_root
from ...capsule.publish import (
    available_targets,
    current_github_login,
    remove_configured_target,
    save_configured_target,
    set_default_target,
)
from ...config.user_settings import load_effective_settings
from ..ui import print_block, print_list_block, print_status


REPO_JSON_SCHEMA = "lelabo.cli.repo/v1"
REPOS_JSON_SCHEMA = "lelabo.cli.repos/v1"


REPO_HELP = """\
Manage advanced publish targets for capsules.

Usage:
  lelabo repo <subcommand> [args]

Subcommands:
  list      List available publish targets for one capsule
  add       Add a GitHub publish target
  use       Set the default publish target
  remove    Remove one configured publish target
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


def _resolve_capsule_root(capsule_ref: str | None, *, caps_dir: Path | None) -> Path:
    if not capsule_ref:
        active = find_active_capsule_root()
        if active is not None:
            return active.resolve()
        direct_children = sorted(child.resolve() for child in Path.cwd().iterdir() if _is_capsule_root(child))
        if len(direct_children) == 1:
            return direct_children[0]
        raise SystemExit("No active capsule found. Pass a capsule path/id or run inside a capsule directory.")
    ref_path = Path(capsule_ref).expanduser()
    if ref_path.exists():
        root = ref_path.resolve()
        if root.is_file():
            root = root.parent
        if _is_capsule_root(root):
            return root
        active = find_active_capsule_root(start=root)
        if active is not None:
            return active.resolve()
        raise SystemExit(f"Path '{capsule_ref}' is not inside a capsule.")
    local_candidate = (Path.cwd() / str(capsule_ref)).resolve()
    if _is_capsule_root(local_candidate):
        return local_candidate
    row = get_capsule(capsule_ref, caps_dir)
    if row is None:
        raise SystemExit(f"Unknown capsule '{capsule_ref}' (not found as path nor stored id/alias).")
    root = Path(str(row.get("path", ""))).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Capsule path does not exist on disk: {root}")
    return root


def _cmd_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo list",
        description="List publish targets for one capsule.",
    )
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir)
    targets = available_targets(capsule_root)
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPOS_JSON_SCHEMA,
                "command": "list",
                "capsule_path": str(capsule_root),
                "targets": targets,
            }
        )
    else:
        if not targets:
            print_status("info", "No publish target is configured for this capsule.")
            return 0
        print_list_block(
            "Publish targets",
            [
                f"{item['name']} | kind: {item.get('kind')} | repo: {item.get('owner', '-')}/{item.get('repo', '-')} | branch: {item.get('branch', '-')} | default: {bool(item.get('default'))} | last_used: {bool(item.get('last_used'))}"
                for item in targets
            ],
        )
    return 0


def _cmd_add(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo add",
        description="Add a GitHub publish target for one capsule.",
    )
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--name", default="github", help="Target name")
    parser.add_argument("--owner", default=None, help="GitHub owner")
    parser.add_argument("--repo", default=None, help="GitHub repository name")
    parser.add_argument("--branch", default=None, help="Target branch")
    parser.add_argument("--path", default=None, help="Capsule path inside the target repo")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="Target repo visibility when created")
    vis.add_argument("--private", action="store_true", help="Target repo visibility when created")
    parser.add_argument("--default", action="store_true", help="Make this the default publish target")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir)
    info = current_github_login()
    github_cfg = settings.get("github", {}) if isinstance(settings.get("github", {}), dict) else {}
    owner = str(args.owner or github_cfg.get("owner") or info or "").strip()
    if not owner:
        raise SystemExit("GitHub owner is required. Set --owner or `lelabo config set github.owner <owner>`.")
    capsule_id = capsule_root.name
    target = save_configured_target(
        capsule_root,
        {
            "name": str(args.name).strip() or "github",
            "kind": "github",
            "owner": owner,
            "repo": str(args.repo or capsule_id).strip() or capsule_id,
            "branch": str(args.branch or github_cfg.get("default_branch", "main")).strip() or "main",
            "path": str(args.path or capsule_id).strip() or capsule_id,
            "visibility": "public"
            if bool(args.public)
            else "private"
            if bool(args.private)
            else str(github_cfg.get("default_visibility", "private")).strip().lower()
            or "private",
        },
        make_default=bool(args.default),
    )
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPO_JSON_SCHEMA,
                "command": "add",
                "capsule_path": str(capsule_root),
                "target": target,
            }
        )
    else:
        print_status("success", "Publish target added.")
        print_block(
            "Target",
            (
                ("name", target.get("name")),
                ("kind", target.get("kind")),
                ("owner", target.get("owner")),
                ("repo", target.get("repo")),
                ("branch", target.get("branch")),
                ("path", target.get("path")),
                ("default", target.get("default")),
            ),
        )
    return 0


def _cmd_use(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo use",
        description="Set the default publish target for one capsule.",
    )
    parser.add_argument("name", help="Target name")
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir)
    try:
        target = set_default_target(capsule_root, args.name)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPO_JSON_SCHEMA,
                "command": "use",
                "capsule_path": str(capsule_root),
                "target": target,
            }
        )
    else:
        print_status("success", "Default publish target updated.")
        print_block("Target", (("name", target.get("name")), ("default", True)))
    return 0


def _cmd_remove(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo remove",
        description="Remove one configured publish target.",
    )
    parser.add_argument("name", help="Target name")
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir)
    try:
        target = remove_configured_target(capsule_root, args.name)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPO_JSON_SCHEMA,
                "command": "remove",
                "capsule_path": str(capsule_root),
                "target": target,
            }
        )
    else:
        print_status("success", "Publish target removed.")
        print_block("Target", (("name", target.get("name")),))
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(REPO_HELP)
        return 0
    cmd = str(args[0]).strip().lower()
    rest = args[1:]
    if cmd == "list":
        return _cmd_list(rest)
    if cmd == "add":
        return _cmd_add(rest)
    if cmd == "use":
        return _cmd_use(rest)
    if cmd == "remove":
        return _cmd_remove(rest)
    raise SystemExit(
        f"Unknown repo subcommand: {cmd}\n\n"
        "Use one of: list, add, use, remove.\n"
        "Run `lelabo repo -h` for usage."
    )


__all__ = ["main"]
