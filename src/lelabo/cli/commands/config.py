"""CLI command for LeLabo user settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ...config.user_settings import (
    DEFAULT_USER_SETTINGS,
    edit_settings_file,
    find_local_override,
    get_key,
    load_effective_settings,
    load_settings_file,
    remove_key,
    set_key,
    user_config_path,
    write_settings,
    dumps_toml,
)
from ..ui import print_block, print_status


CONFIG_HELP = """\
Manage LeLabo user settings.

Usage:
  lelabo config <subcommand> [args]

Subcommands:
  path      Show global and local config paths
  show      Show effective settings
  get       Read one dotted key (e.g. github.owner)
  set       Write one dotted key into a config file
  reset     Reset one dotted key or a whole config file to defaults
  edit      Open config file in $EDITOR

Help:
  lelabo config -h
  lelabo config <subcommand> -h
"""


def _target_file(raw_file: str | None) -> Path:
    if raw_file:
        return Path(raw_file).expanduser().resolve()
    return user_config_path()


def _is_interactive_tty() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _confirm(question: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return bool(default)
    return answer in {"y", "yes"}


def _cmd_path(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo config path", description="Show config file paths.")
    args = parser.parse_args(argv)
    _ = args
    global_path = user_config_path()
    local_path = find_local_override()
    print_block(
        "Config paths",
        (
            ("global", global_path),
            ("local", local_path if local_path is not None else "-"),
        ),
    )
    return 0


def _cmd_show(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo config show", description="Show user settings.")
    parser.add_argument("--file", default=None, help="Read this config file instead of effective global+local settings.")
    args = parser.parse_args(argv)
    if args.file:
        data = load_settings_file(Path(args.file))
    else:
        data = load_effective_settings()
    print(dumps_toml(data), end="")
    return 0


def _cmd_get(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo config get", description="Read one config value.")
    parser.add_argument("key", help="Dotted key, e.g. github.owner")
    parser.add_argument("--file", default=None, help="Read this config file instead of effective global+local settings.")
    args = parser.parse_args(argv)
    data = load_settings_file(Path(args.file)) if args.file else load_effective_settings()
    value = get_key(data, args.key)
    print(value if value is not None else "")
    return 0


def _cmd_set(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo config set", description="Set one config value.")
    parser.add_argument("key", help="Dotted key, e.g. github.owner")
    parser.add_argument("value", help="Raw value, parsed as bool/string based on key")
    parser.add_argument("--file", default=None, help="Target config file (default: global user config).")
    args = parser.parse_args(argv)

    path = _target_file(args.file)
    data = load_settings_file(path)
    updated = set_key(data, args.key, args.value)
    write_settings(path, updated)
    print_status("success", "Config updated.")
    print_block(
        "Config change",
        (
            ("file", path),
            ("key", args.key),
            ("value", get_key(updated, args.key)),
        ),
    )
    return 0


def _cmd_edit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo config edit", description="Open the config file in $EDITOR.")
    parser.add_argument("--file", default=None, help="Target config file (default: global user config).")
    args = parser.parse_args(argv)
    path = _target_file(args.file)
    return int(edit_settings_file(path))


def _cmd_reset(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo config reset",
        description="Reset one config key or one whole config file to defaults.",
    )
    parser.add_argument("key", nargs="?", default=None, help="Optional dotted key to reset, e.g. github.owner")
    parser.add_argument("--file", default=None, help="Target config file (default: global user config).")
    parser.add_argument("--yes", action="store_true", help="Accept full-file reset non-interactively")
    args = parser.parse_args(argv)

    path = _target_file(args.file)
    if args.key:
        data = load_settings_file(path)
        updated = remove_key(data, args.key)
        write_settings(path, updated)
        print_status("success", "Config key reset.")
        print_block(
            "Config reset",
            (
                ("file", path),
                ("key", args.key),
                ("value", get_key(updated, args.key)),
            ),
        )
        return 0

    if not args.yes:
        if not _is_interactive_tty():
            raise SystemExit("Refusing to reset the whole config file non-interactively without `--yes`.")
        if not _confirm(f"Reset config file {path} to defaults?", default=False):
            raise SystemExit("Config reset canceled by user.")
    write_settings(path, dict(DEFAULT_USER_SETTINGS))
    print_status("success", "Config reset to defaults.")
    print_block("Config reset", (("file", path),))
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(CONFIG_HELP)
        return 0

    cmd = str(args[0]).strip().lower()
    rest = args[1:]
    if cmd == "path":
        return _cmd_path(rest)
    if cmd == "show":
        return _cmd_show(rest)
    if cmd == "get":
        return _cmd_get(rest)
    if cmd == "set":
        return _cmd_set(rest)
    if cmd == "reset":
        return _cmd_reset(rest)
    if cmd == "edit":
        return _cmd_edit(rest)
    raise SystemExit(
        f"Unknown config subcommand: {cmd}\n\n"
        "Use one of: path, show, get, set, reset, edit.\n"
        "Run `lelabo config -h` for usage."
    )
