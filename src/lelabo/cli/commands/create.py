from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ...capsule import create_capsule_scaffold


CREATE_HELP = """\
Create local LeLabo scaffolds.

Usage:
  lelabo create <target> [args]

Targets:
  capsule    Create a capsule folder scaffold in the current directory
             (registered by default in user app-data capsules store)

Help:
  lelabo create -h
  lelabo create <target> -h
"""


def _resolve_capsule_name(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    positional = args.name
    from_flag = args.name_flag

    if positional and from_flag and positional != from_flag:
        parser.error("Provide capsule name only once: positional or --name/-n.")
    name = from_flag or positional
    if not name:
        parser.error("Provide capsule name with positional value or --name/-n.")
    return str(name)


def create_capsule(
    *,
    capsule_name: str,
    base_dir: Path | None = None,
    force: bool = False,
    register: bool = True,
    alias: str | None = None,
    capsules_dir: Path | None = None,
) -> Path:
    return create_capsule_scaffold(
        capsule_name=capsule_name,
        base_dir=base_dir,
        force=force,
        register=register,
        alias=alias,
        capsules_dir=capsules_dir,
    )


def _cmd_capsule(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo create capsule")
    parser.add_argument("name", nargs="?", help="Capsule name (folder name)")
    parser.add_argument("-n", "--name", dest="name_flag", default=None, help="Capsule name (folder name)")
    parser.add_argument("--dir", dest="base_dir", default=".", help="Base directory where the capsule is created")
    parser.add_argument("--alias", default=None, help="Optional alias used in `lelabo capsule list/show`")
    parser.add_argument(
        "--capsules-dir",
        default=None,
        help="Override capsules index/store path (defaults to user app-data location)",
    )
    parser.add_argument("--force", action="store_true", help="Scaffold even if the capsule directory already exists")
    parser.add_argument("--no-register", action="store_true", help="Create files only, without adding to capsule index")
    args = parser.parse_args(argv)

    name = _resolve_capsule_name(parser, args)
    try:
        out = create_capsule(
            capsule_name=name,
            base_dir=Path(args.base_dir),
            force=bool(args.force),
            register=not bool(args.no_register),
            alias=args.alias,
            capsules_dir=Path(args.capsules_dir) if args.capsules_dir else None,
        )
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc))

    print(str(out))
    print()
    print("Next steps:")
    print("  - Start with README.md")
    print("  - If you use Codex/Claude, open AGENTS.md")
    print("  - Exact contracts: resources/LELABO_REFERENCE.md")
    print("  - Param mapping: resources/PARAM_FLOW.md")
    print("  - Update-rule contract: resources/UPDATE_RULE_LIFECYCLE.md")
    print("  - Paper-pack workflow: resources/PAPER_PACK_PLAYBOOK.md")
    print('  - Official first path: uncomment `@register_optimizer("capsule_sgd")` in `optimizers/example.py`')
    print("  - Run `lelabo list optimizers`")
    print("  - Run `lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml`")
    print("  - Run `pytest -q tests`")
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(CREATE_HELP)
        return 0

    cmd = args[0]
    rest = args[1:]

    if cmd == "capsule":
        return _cmd_capsule(rest)

    raise SystemExit(
        f"Unknown create target: {cmd}\n\n"
        "Use one of: capsule.\n"
        "Run `lelabo create -h` for usage."
    )
