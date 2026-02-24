from __future__ import annotations

import sys
from typing import Sequence

from ._common import coerce_system_exit_code, is_help_token
from .commands.audit import main as run_audit_command
from .commands.capsule import main as run_capsule_command
from .commands.create import main as run_create_command
from .commands.list import main as run_list_command
from .commands.train import main as run_train_command


def _run_command(fn, argv: Sequence[str]) -> int:
    try:
        return int(fn(list(argv)))
    except SystemExit as exc:
        return coerce_system_exit_code(exc.code)


def _run_train_cli(argv: Sequence[str]) -> int:
    return _run_command(run_train_command, argv)


def _run_audit_cli(argv: Sequence[str]) -> int:
    return _run_command(run_audit_command, argv)


def _run_capsule_cli(argv: Sequence[str]) -> int:
    return _run_command(run_capsule_command, argv)


def _run_create_cli(argv: Sequence[str]) -> int:
    return _run_command(run_create_command, argv)


def _run_list_cli(argv: Sequence[str]) -> int:
    return _run_command(run_list_command, argv)


ROOT_HELP = """\
LeLabo command-line interface.

Usage:
  lelabo <command> [args]

Commands:
  train      Run supervised or RL training (explicit mode)
  audit      Run update-rule audit tests
  create     Create local LeLabo scaffolds
  list       List available update-rules/datasets/models/optimizers/metrics/schedulers
  capsule    Pack/install/store/restore/list/show/rerun experiment capsules

Help:
  lelabo -h
  lelabo train -h
  lelabo audit -h
  lelabo create -h
  lelabo list -h
  lelabo capsule -h

Examples:
  lelabo train supervised --dataset iris --model mlp --algo bp
  lelabo train rl --env CartPole-v1 --rl-algo ppo
  lelabo audit --all --modes supervised,rl
  lelabo create capsule --name my_capsule
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

    if cmd == "create":
        if not rest or is_help_token(rest[0]):
            return _run_create_cli(["--help"])
        return _run_create_cli(rest)

    if cmd in {"list", "ls"}:
        if rest and is_help_token(rest[0]):
            return _run_list_cli(["--help"])
        return _run_list_cli(rest)

    if cmd == "capsule":
        if not rest or is_help_token(rest[0]):
            return _run_capsule_cli(["--help"])
        return _run_capsule_cli(rest)

    raise SystemExit(
        "Unknown command: "
        f"{cmd}\n\n"
        "Use one of: train, audit, create, list, capsule.\n"
        "Run `lelabo --help` for usage."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
