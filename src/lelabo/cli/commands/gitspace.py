"""CLI command for gitspace lifecycle and inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ...capsule import add_capsule_to_gitspace, find_gitspace_root, init_gitspace, load_gitspace
from ..ui import print_block, print_list_block, print_status


GITSPACE_HELP = """\
Manage LeLabo gitspaces.

Usage:
  lelabo gitspace <subcommand> [args]

Subcommands:
  init      Create a `.lelabo/gitspace.toml` manifest
  show      Show the detected gitspace
  list      List capsules declared in the gitspace
  add       Add an existing capsule to the gitspace manifest

Help:
  lelabo gitspace -h
  lelabo gitspace <subcommand> -h
"""

GITSPACE_JSON_SCHEMA = "lelabo.cli.gitspace/v1"
GITSPACES_JSON_SCHEMA = "lelabo.cli.gitspaces/v1"


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _resolve_gitspace_root(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    resolved = find_gitspace_root()
    if resolved is not None:
        return resolved
    return Path.cwd().resolve()


def _cmd_init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo gitspace init",
        description="Create a LeLabo gitspace manifest in PATH or the current directory.",
    )
    parser.add_argument("path", nargs="?", default=".", help="Gitspace root directory")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    gitspace = init_gitspace(Path(args.path))
    if bool(args.json):
        _print_json(
            {
                "schema_version": GITSPACE_JSON_SCHEMA,
                "command": "init",
                "gitspace": {
                    "name": gitspace["name"],
                    "root": gitspace["root"],
                    "manifest_path": gitspace["manifest_path"],
                },
            }
        )
    else:
        print_status("success", "Gitspace initialized.")
        print_block(
            "Gitspace",
            (
                ("name", gitspace["name"]),
                ("root", gitspace["root"]),
                ("manifest_path", gitspace["manifest_path"]),
            ),
        )
    return 0


def _cmd_show(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo gitspace show",
        description="Show the detected LeLabo gitspace.",
    )
    parser.add_argument("path", nargs="?", default=None, help="Path inside the gitspace (defaults to cwd)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        gitspace = load_gitspace(_resolve_gitspace_root(args.path))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    if bool(args.json):
        _print_json(
            {
                "schema_version": GITSPACE_JSON_SCHEMA,
                "command": "show",
                "gitspace": gitspace,
            }
        )
    else:
        print_block(
            "Gitspace",
            (
                ("name", gitspace["name"]),
                ("root", gitspace["root"]),
                ("manifest_path", gitspace["manifest_path"]),
                ("capsules", len(gitspace["capsules"])),
            ),
        )
    return 0


def _cmd_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo gitspace list",
        description="List capsules declared in the detected LeLabo gitspace.",
    )
    parser.add_argument("path", nargs="?", default=None, help="Path inside the gitspace (defaults to cwd)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        gitspace = load_gitspace(_resolve_gitspace_root(args.path))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    if bool(args.json):
        _print_json(
            {
                "schema_version": GITSPACES_JSON_SCHEMA,
                "command": "list",
                "gitspace": {
                    "name": gitspace["name"],
                    "root": gitspace["root"],
                },
                "capsules": list(gitspace["capsules"]),
            }
        )
    else:
        if not gitspace["capsules"]:
            print_status("info", "This gitspace does not declare any capsules yet.")
            return 0
        print_list_block(
            "Gitspace capsules",
            [f"{item['id']} | path: {item['path']}" for item in gitspace["capsules"]],
        )
    return 0


def _cmd_add(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo gitspace add",
        description="Add an existing capsule to the gitspace manifest without moving files.",
    )
    parser.add_argument("capsule_path", help="Path to a capsule already inside the gitspace")
    parser.add_argument("path", nargs="?", default=None, help="Gitspace root (defaults to detected gitspace)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    target_root = _resolve_gitspace_root(args.path)
    try:
        gitspace = add_capsule_to_gitspace(Path(args.capsule_path), gitspace_root=target_root)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    if bool(args.json):
        _print_json(
            {
                "schema_version": GITSPACE_JSON_SCHEMA,
                "command": "add",
                "gitspace": {
                    "name": gitspace["name"],
                    "root": gitspace["root"],
                    "manifest_path": gitspace["manifest_path"],
                },
                "result": {
                    "action": gitspace["action"],
                    "capsule": gitspace["capsule"],
                },
            }
        )
    else:
        print_status("success", "Capsule added to gitspace.")
        print_block(
            "Gitspace",
            (
                ("name", gitspace["name"]),
                ("root", gitspace["root"]),
                ("capsule_id", gitspace["capsule"]["id"]),
                ("capsule_path", gitspace["capsule"]["path"]),
                ("action", gitspace["action"]),
            ),
        )
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(GITSPACE_HELP)
        return 0

    cmd = str(args[0]).strip().lower()
    rest = args[1:]
    if cmd == "init":
        return _cmd_init(rest)
    if cmd == "show":
        return _cmd_show(rest)
    if cmd == "list":
        return _cmd_list(rest)
    if cmd == "add":
        return _cmd_add(rest)
    raise SystemExit(
        f"Unknown gitspace subcommand: {cmd}\n\n"
        "Use one of: init, show, list, add.\n"
        "Run `lelabo gitspace -h` for usage."
    )
