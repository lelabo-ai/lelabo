from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ...api.capsule import install_capsule
from ...api.capsule import list_capsules
from ...api.capsule import pack_capsule
from ...api.capsule import remove_capsule
from ...api.capsule import restore_capsule
from ...api.capsule import rerun_capsule
from ...api.capsule import show_capsule
from ...api.capsule import store_capsule


CAPSULE_HELP = """\
Manage LeLabo experiment capsules.

Usage:
  lelabo capsule <subcommand> [args]

Subcommands:
  pack       Build a shareable capsule bundle from a run/sweep/config
  install    Install a capsule bundle into the local capsule store
  store      Store a local capsule directory into the capsule library/cache
  restore    Move an installed capsule out of cache into a local working directory
  list       List installed capsules
  show       Show one installed capsule entry
  remove     Remove one installed capsule entry (and files by default)
  rerun      Rerun a capsule entrypoint

Help:
  lelabo capsule -h
  lelabo capsule <subcommand> -h
"""


def _cmd_pack(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo capsule pack")
    parser.add_argument("--from", dest="source", required=True, help="Source run dir / sweep dir / config file")
    parser.add_argument("--out", dest="out_path", default=None, help="Output bundle path (.tar.gz or .tar.zst)")
    parser.add_argument("--id", dest="capsule_id", default=None, help="Optional capsule id")
    parser.add_argument("--with-code-snapshot", action="store_true", help="Embed src/ snapshot in capsule")
    args = parser.parse_args(argv)

    out = pack_capsule(
        source=Path(args.source),
        out_path=Path(args.out_path) if args.out_path else None,
        capsule_id=args.capsule_id,
        include_code_snapshot=bool(args.with_code_snapshot),
    )
    print(str(out))
    return 0


def _cmd_install(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo capsule install")
    parser.add_argument("bundle", help="Path to capsule bundle (.tar.gz/.tar.zst)")
    parser.add_argument("--name", dest="alias", default=None, help="Optional alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    args = parser.parse_args(argv)

    entry = install_capsule(
        bundle_path=Path(args.bundle),
        alias=args.alias,
        capsules_dir=Path(args.capsules_dir) if args.capsules_dir else None,
    )
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    return 0


def _cmd_store(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo capsule store")
    parser.add_argument(
        "-n",
        "--name",
        dest="alias",
        required=True,
        help="Capsule alias to register in the library",
    )
    parser.add_argument(
        "--from",
        dest="source",
        default=None,
        help="Path inside a capsule directory (defaults to current working directory)",
    )
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    args = parser.parse_args(argv)

    try:
        entry = store_capsule(
            alias=args.alias,
            source_path=Path(args.source) if args.source else None,
            capsules_dir=Path(args.capsules_dir) if args.capsules_dir else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    print(json.dumps(entry, indent=2, ensure_ascii=False))
    return 0


def _cmd_restore(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo capsule restore")
    parser.add_argument("id_or_alias", help="Installed capsule id or alias to move out of cache")
    parser.add_argument("--to", dest="destination_dir", default=None, help="Target directory (defaults to cwd)")
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Optional destination folder name (defaults to capsule id)",
    )
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    args = parser.parse_args(argv)

    try:
        restored = restore_capsule(
            capsule_or_alias=args.id_or_alias,
            destination_dir=Path(args.destination_dir) if args.destination_dir else None,
            name=args.name,
            capsules_dir=Path(args.capsules_dir) if args.capsules_dir else None,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    print(json.dumps(restored, indent=2, ensure_ascii=False))
    return 0


def _cmd_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo capsule list")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    args = parser.parse_args(argv)

    rows = list_capsules(Path(args.capsules_dir) if args.capsules_dir else None)
    for row in rows:
        aliases = ",".join(row.get("aliases", [])) or "-"
        print(f"{row.get('capsule_id')}\t{aliases}\t{row.get('installed_at')}\t{row.get('path')}")
    if not rows:
        print("(no capsules installed)")
    return 0


def _cmd_show(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo capsule show")
    parser.add_argument("id_or_alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    args = parser.parse_args(argv)

    try:
        row = show_capsule(args.id_or_alias, Path(args.capsules_dir) if args.capsules_dir else None)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return 0


def _cmd_rerun(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo capsule rerun")
    parser.add_argument("id_or_alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    parser.add_argument("--env", dest="env_mode", default="current", choices=["current", "venv"])
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Extra args appended to replay command")
    args = parser.parse_args(argv)

    extra = list(args.args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return int(
        rerun_capsule(
            capsule_or_alias=args.id_or_alias,
            capsules_dir=Path(args.capsules_dir) if args.capsules_dir else None,
            env_mode=args.env_mode,
            extra_args=extra,
        )
    )


def _cmd_remove(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo capsule remove")
    parser.add_argument("id_or_alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Only remove from capsule registry, keep capsule files on disk",
    )
    parser.add_argument(
        "-r",
        action="store_true",
        dest="rm_recursive",
        help="Used with -f as '-rf' to allow deleting capsule files outside cache.",
    )
    parser.add_argument(
        "-f",
        action="store_true",
        dest="rm_force",
        help="Used with -r as '-rf' to allow deleting capsule files outside cache.",
    )
    parser.add_argument(
        "--force-external-delete",
        action="store_true",
        help="Allow deleting capsule files even when stored outside capsules cache.",
    )
    args = parser.parse_args(argv)
    allow_external_delete = bool(args.force_external_delete or (args.rm_recursive and args.rm_force))
    if (args.rm_recursive or args.rm_force) and not allow_external_delete:
        raise SystemExit("Use '-rf' together to allow external capsule deletion.")

    try:
        removed = remove_capsule(
            capsule_or_alias=args.id_or_alias,
            capsules_dir=Path(args.capsules_dir) if args.capsules_dir else None,
            delete_files=not bool(args.keep_files),
            allow_external_delete=allow_external_delete,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    print(json.dumps(removed, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(CAPSULE_HELP)
        return 0

    cmd = args[0]
    rest = args[1:]

    if cmd == "pack":
        return _cmd_pack(rest)
    if cmd == "install":
        return _cmd_install(rest)
    if cmd == "store":
        return _cmd_store(rest)
    if cmd == "restore":
        return _cmd_restore(rest)
    if cmd == "list":
        return _cmd_list(rest)
    if cmd == "show":
        return _cmd_show(rest)
    if cmd == "remove":
        return _cmd_remove(rest)
    if cmd == "rerun":
        return _cmd_rerun(rest)

    raise SystemExit(
        f"Unknown capsule subcommand: {cmd}\n\n"
        "Use one of: pack, install, store, restore, list, show, remove, rerun.\n"
        "Run `lelabo capsule -h` for usage."
    )
