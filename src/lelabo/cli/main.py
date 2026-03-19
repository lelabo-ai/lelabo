"""Top-level dispatcher for the LeLabo command-line interface."""

from __future__ import annotations

import sys
from typing import Sequence

from ._common import CLI_USER_ERROR_TYPES, coerce_system_exit_code, is_help_token
from .commands.audit import main as run_audit_command
from .commands.capsule import main as run_capsule_command
from .commands.config import main as run_config_command
from .commands.list import main as run_list_command
from .commands.push import main as run_push_command
from .commands.repo import main as run_repo_command
from .commands.sweep import main as run_sweep_command
from .commands.train import main as run_train_command


def _run_command(fn, argv: Sequence[str]) -> int:
    try:
        return int(fn(list(argv)))
    except SystemExit as exc:
        return coerce_system_exit_code(exc.code)
    except CLI_USER_ERROR_TYPES as exc:
        return coerce_system_exit_code(str(exc))


def _run_train_cli(argv: Sequence[str]) -> int:
    return _run_command(run_train_command, argv)


def _run_audit_cli(argv: Sequence[str]) -> int:
    return _run_command(run_audit_command, argv)


def _run_capsule_cli(argv: Sequence[str]) -> int:
    return _run_command(run_capsule_command, argv)

def _run_config_cli(argv: Sequence[str]) -> int:
    return _run_command(run_config_command, argv)

def _run_list_cli(argv: Sequence[str]) -> int:
    return _run_command(run_list_command, argv)

def _run_push_cli(argv: Sequence[str]) -> int:
    return _run_command(run_push_command, argv)

def _run_repo_cli(argv: Sequence[str]) -> int:
    return _run_command(run_repo_command, argv)


def _run_sweep_cli(argv: Sequence[str]) -> int:
    return _run_command(run_sweep_command, argv)


ROOT_HELP = """\
LeLabo command-line interface.

Usage:
  lelabo <command> [args]

Commands:
  train      Run supervised or RL training
  sweep      Discover and run workspace sweeps
  audit      Run update-rule audit checks
  push       Publish a capsule to a repo target
  repo       Manage advanced capsule publish targets
  config     Manage user settings and defaults
  list       List available registries and active-capsule exports
  capsule    Create, install, store, inspect, and share capsules

Help:
  lelabo -h
  lelabo train -h
  lelabo audit -h
  lelabo push -h
  lelabo repo -h
  lelabo config -h
  lelabo list -h
  lelabo capsule -h

Examples:
  lelabo train supervised --dataset iris --model mlp --rule bp
  lelabo train rl --env CartPole-v1 --algo ppo
  lelabo audit --all --modes supervised,rl
  lelabo push my_capsule
  lelabo repo add my_capsule --name public --owner your-org --repo method-zoo
  lelabo config set github.owner your-org
  lelabo capsule init my_capsule
  lelabo list update-rules
  lelabo capsule pack --from outputs/runs/demo/run1
"""


def _print_root_help() -> None:
    print(ROOT_HELP)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_root_help()
        return 0

    cmd = str(argv[0]).strip().lower()
    rest = list(argv[1:])

    if is_help_token(cmd):
        _print_root_help()
        return 0

    if cmd == "train":
        if not rest or is_help_token(rest[0]):
            return _run_train_cli(["-h"])
        return _run_train_cli(rest)

    if cmd == "audit":
        if rest and is_help_token(rest[0]):
            return _run_audit_cli(["-h"])
        return _run_audit_cli(rest)

    if cmd == "push":
        if rest and is_help_token(rest[0]):
            return _run_push_cli(["-h"])
        return _run_push_cli(rest)

    if cmd == "repo":
        if not rest or is_help_token(rest[0]):
            return _run_repo_cli(["-h"])
        return _run_repo_cli(rest)

    if cmd == "config":
        if not rest or is_help_token(rest[0]):
            return _run_config_cli(["-h"])
        return _run_config_cli(rest)

    if cmd == "gitspace":
        raise SystemExit("`lelabo gitspace` has been removed. Use `lelabo push` or `lelabo repo`.")

    if cmd in {"list", "ls"}:
        if rest and is_help_token(rest[0]):
            return _run_list_cli(["--help"])
        return _run_list_cli(rest)

    if cmd == "sweep":
        if rest and is_help_token(rest[0]):
            return _run_sweep_cli(["--help"])
        return _run_sweep_cli(rest)

    if cmd == "capsule":
        if not rest or is_help_token(rest[0]):
            return _run_capsule_cli(["--help"])
        return _run_capsule_cli(rest)

    raise SystemExit(
        "Unknown command: "
        f"{cmd}\n\n"
        "Use one of: train, sweep, audit, push, repo, config, list, capsule.\n"
        "Run `lelabo --help` for usage."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
