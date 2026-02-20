from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..capsule.install import install_capsule
from ..capsule.pack import pack_capsule
from ..capsule.registry import get_capsule, list_capsules
from ..capsule.rerun import rerun_capsule


def _cmd_pack(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="lelabo capsule pack")
    p.add_argument("--from", dest="source", required=True, help="Source run dir / sweep dir / config file")
    p.add_argument("--out", dest="out_path", default=None, help="Output bundle path (.tar.gz or .tar.zst)")
    p.add_argument("--id", dest="capsule_id", default=None, help="Optional capsule id")
    p.add_argument("--with-code-snapshot", action="store_true", help="Embed src/ snapshot in capsule")
    args = p.parse_args(argv)

    out = pack_capsule(
        source=Path(args.source),
        out_path=Path(args.out_path) if args.out_path else None,
        capsule_id=args.capsule_id,
        include_code_snapshot=bool(args.with_code_snapshot),
    )
    print(str(out))
    return 0


def _cmd_install(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="lelabo capsule install")
    p.add_argument("bundle", help="Path to capsule bundle (.tar.gz/.tar.zst)")
    p.add_argument("--name", dest="alias", default=None, help="Optional alias")
    p.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    args = p.parse_args(argv)

    entry = install_capsule(
        bundle_path=Path(args.bundle),
        alias=args.alias,
        capsules_dir=Path(args.capsules_dir) if args.capsules_dir else None,
    )
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    return 0


def _cmd_list(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="lelabo capsule list")
    p.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    args = p.parse_args(argv)

    rows = list_capsules(Path(args.capsules_dir) if args.capsules_dir else None)
    for r in rows:
        aliases = ",".join(r.get("aliases", [])) or "-"
        print(f"{r.get('capsule_id')}\t{aliases}\t{r.get('installed_at')}\t{r.get('path')}")
    if not rows:
        print("(no capsules installed)")
    return 0


def _cmd_show(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="lelabo capsule show")
    p.add_argument("id_or_alias")
    p.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    args = p.parse_args(argv)

    row = get_capsule(args.id_or_alias, Path(args.capsules_dir) if args.capsules_dir else None)
    if row is None:
        raise SystemExit(f"Unknown capsule '{args.id_or_alias}'")
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return 0


def _cmd_rerun(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="lelabo capsule rerun")
    p.add_argument("id_or_alias")
    p.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    p.add_argument("--env", dest="env_mode", default="current", choices=["current", "venv"])
    p.add_argument("args", nargs=argparse.REMAINDER, help="Extra args appended to replay command")
    args = p.parse_args(argv)

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


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("Usage: lelabo capsule <pack|install|list|show|rerun> ...")

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "pack":
        return _cmd_pack(rest)
    if cmd == "install":
        return _cmd_install(rest)
    if cmd == "list":
        return _cmd_list(rest)
    if cmd == "show":
        return _cmd_show(rest)
    if cmd == "rerun":
        return _cmd_rerun(rest)

    raise SystemExit(f"Unknown capsule subcommand: {cmd}")
