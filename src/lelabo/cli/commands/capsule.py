from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ...capsule import (
    checkout_capsule,
    create_capsule_scaffold,
    get_capsule,
    install_capsule,
    list_capsules,
    pack_capsule,
    remove_capsule,
    stash_capsule,
)
from ...capsule.plugins.discovery import find_active_capsule_root


CAPSULE_HELP = """\
Manage LeLabo experiment capsules.

Usage:
  lelabo capsule <subcommand> [args]

Subcommands:
  init       Create a local work capsule in the current workspace
  stash      Move a local capsule into the local capsule store/cache
  checkout   Move a stored capsule back into a local workspace
  install    Import an external capsule bundle into the local capsule store
  pack       Build a shareable capsule bundle from a run/sweep/config
  list       List stored capsules
  show       Show one stored capsule entry
  remove     Remove one stored capsule entry (and files by default)

Help:
  lelabo capsule -h
  lelabo capsule <subcommand> -h
"""

CAPSULE_JSON_SCHEMA = "lelabo.cli.capsule/v1"
CAPSULES_JSON_SCHEMA = "lelabo.cli.capsules/v1"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _print_json(payload: Any) -> None:
    print(_json_dumps(payload))


def _render_aliases(row: dict[str, Any]) -> str:
    aliases = [str(item).strip() for item in list(row.get("aliases", []) or []) if str(item).strip()]
    return ", ".join(aliases) if aliases else "-"


def _capsule_json_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "capsule_id": row.get("capsule_id"),
        "aliases": list(row.get("aliases", []) or []),
        "path": row.get("path"),
        "kind": row.get("kind"),
    }


def _capsule_command_json(command: str, row: dict[str, Any], *, result_fields: Sequence[str] = ()) -> dict[str, Any]:
    result = {field: row[field] for field in result_fields if field in row}
    payload: dict[str, Any] = {
        "schema_version": CAPSULE_JSON_SCHEMA,
        "command": command,
        "capsule": _capsule_json_entry(row),
    }
    if result:
        payload["result"] = result
    return payload


def _print_entry_block(title: str, row: dict[str, Any]) -> None:
    print(title)
    print(f"capsule_id: {row.get('capsule_id', '-')}")
    print(f"aliases: {_render_aliases(row)}")
    if str(row.get("path", "")).strip():
        print(f"path: {row.get('path')}")


def _print_action_block(title: str, row: dict[str, Any], *, extra_fields: Sequence[str]) -> None:
    _print_entry_block(title, row)
    for field in extra_fields:
        if field not in row:
            continue
        print(f"{field}: {row[field]}")


def _is_capsule_root(path: Path) -> bool:
    return path.is_dir() and ((path / "capsule.toml").is_file() or (path / "manifest.json").is_file())


def _collect_child_capsule_roots(container: Path) -> list[Path]:
    if not container.exists():
        raise FileNotFoundError(f"Capsule container path not found: {container}")
    if not container.is_dir():
        raise ValueError(f"Capsule container path is not a directory: {container}")
    if _is_capsule_root(container):
        raise ValueError(f"`lelabo capsule stash --all` expects a container directory, not a capsule root: {container}")

    roots = sorted(child.resolve() for child in container.iterdir() if _is_capsule_root(child))
    if not roots:
        raise ValueError(f"No direct child capsule folders found in '{container}'.")
    return roots


def _cmd_init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule init",
        description="Create a new local capsule scaffold in your current workspace.",
        epilog="Examples:\n  lelabo capsule init my_capsule\n  lelabo capsule init my_paper --dir workspaces/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("name", help="Capsule name (folder name)")
    parser.add_argument("--dir", dest="base_dir", default=".", help="Parent directory where the capsule is created")
    parser.add_argument("--force", action="store_true", help="Create scaffold even if the target directory already exists")
    args = parser.parse_args(argv)

    try:
        out = create_capsule_scaffold(
            capsule_name=str(args.name),
            base_dir=Path(args.base_dir),
            force=bool(args.force),
            register=False,
        )
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc))

    print(str(out))
    print()
    print("Next steps:")
    print("  - Start with README.md")
    print("  - If you use Codex/Claude, open AGENTS.md")
    print("  - How to add X: resources/EXTENSION_RECIPES.md")
    print("  - Exact contracts: resources/LELABO_REFERENCE.md")
    print("  - Param mapping: resources/PARAM_FLOW.md")
    print("  - Update-rule contract: resources/UPDATE_RULE_LIFECYCLE.md")
    print("  - Paper-pack workflow: resources/PAPER_PACK_PLAYBOOK.md")
    print('  - Official first path: uncomment `@register_optimizer("capsule_sgd")` in `optimizers/example.py`')
    print("  - Run `lelabo list optimizers`")
    print("  - Run `lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml`")
    print("  - Run `pytest -q tests`")
    return 0


def _cmd_pack(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule pack",
        description="Build a shareable capsule bundle from a run directory, sweep directory, or config file.",
        epilog="Examples:\n  lelabo capsule pack --from runs/exp1\n  lelabo capsule pack --from outputs/sweeps/demo --out demo_capsule.tar.gz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser = argparse.ArgumentParser(
        prog="lelabo capsule install",
        description="Import an external capsule bundle into the local capsule store.",
        epilog="Examples:\n  lelabo capsule install demo_capsule.tar.gz\n  lelabo capsule install demo_capsule.tar.gz --alias demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bundle", help="Path to capsule bundle (.tar.gz/.tar.zst)")
    parser.add_argument("--alias", default=None, help="Optional alias inside the capsule store")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    entry = install_capsule(
        bundle_path=Path(args.bundle),
        alias=args.alias,
        capsules_dir=Path(args.capsules_dir) if args.capsules_dir else None,
    )
    if bool(args.json):
        _print_json(
            _capsule_command_json(
                "install",
                entry,
                result_fields=("installed_at", "source_bundle"),
            )
        )
    else:
        _print_action_block("installed capsule into store:", entry, extra_fields=("installed_at", "source_bundle"))
    return 0


def _cmd_stash(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule stash",
        description="Move a local capsule into the local capsule store.",
        epilog=(
            "Examples:\n"
            "  lelabo capsule stash\n"
            "  lelabo capsule stash ./my_capsule --alias paper_demo\n"
            "  lelabo capsule stash --all ./workspace_capsules"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", nargs="?", default=None, help="Capsule root to stash (defaults to active capsule from cwd)")
    parser.add_argument("--alias", default=None, help="Optional alias inside the capsule store")
    parser.add_argument("--all", action="store_true", help="Stash all direct child capsule folders from SOURCE or '.'")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    caps_dir = Path(args.capsules_dir) if args.capsules_dir else None
    source = Path(args.source).expanduser() if args.source else None

    try:
        if bool(args.all):
            if args.alias is not None:
                raise ValueError("`--alias` cannot be used with `lelabo capsule stash --all`.")
            container = source.resolve() if source is not None else Path(".").resolve()
            if source is None:
                active_root = find_active_capsule_root(start=container)
                if active_root is not None:
                    raise ValueError(
                        "`lelabo capsule stash --all` expects a container directory. "
                        "Run it from a parent folder or pass SOURCE explicitly."
                    )
            roots = _collect_child_capsule_roots(container)
            stored = [stash_capsule(source_path=root, capsules_dir=caps_dir) for root in roots]
            payload = {
                "schema_version": CAPSULES_JSON_SCHEMA,
                "command": "stash",
                "count": len(stored),
                "capsules": [
                    {
                        **_capsule_json_entry(row),
                        "result": {field: row[field] for field in ("stored_from", "moved") if field in row},
                    }
                    for row in stored
                ],
            }
            if bool(args.json):
                _print_json(payload)
            else:
                print(f"stashed {len(stored)} capsules:")
                for row in stored:
                    print(f"- {row.get('capsule_id')} -> {row.get('path')}")
            return 0

        entry = stash_capsule(
            alias=args.alias,
            source_path=source,
            capsules_dir=caps_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    if bool(args.json):
        _print_json(_capsule_command_json("stash", entry, result_fields=("stored_from", "moved")))
    else:
        _print_action_block("stashed capsule:", entry, extra_fields=("stored_from", "moved"))
    return 0


def _cmd_checkout(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule checkout",
        description="Move a stored capsule back into a local workspace.",
        epilog="Examples:\n  lelabo capsule checkout my_capsule\n  lelabo capsule checkout my_capsule ./workbench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("id_or_alias", help="Stored capsule id or alias to move back into a local workspace")
    parser.add_argument("destination", nargs="?", default=".", help="Parent directory where the capsule folder is recreated")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    caps_dir = Path(args.capsules_dir) if args.capsules_dir else None

    try:
        row = get_capsule(args.id_or_alias, caps_dir)
        if row is None:
            raise ValueError(f"Unknown capsule '{args.id_or_alias}'")
        restored = checkout_capsule(
            capsule_or_alias=args.id_or_alias,
            destination_dir=Path(args.destination),
            capsules_dir=caps_dir,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    if bool(args.json):
        json_row = dict(row)
        json_row.update(restored)
        json_row["path"] = restored.get("checked_out_path")
        _print_json(
            _capsule_command_json(
                "checkout",
                json_row,
                result_fields=("source_path", "checked_out_path", "removed_from_cache", "moved"),
            )
        )
    else:
        print("checked out capsule:")
        print(f"capsule_id: {restored.get('capsule_id', '-')}")
        print(f"source_path: {restored.get('source_path', '-')}")
        print(f"checked_out_path: {restored.get('checked_out_path', '-')}")
        print(f"removed_from_cache: {restored.get('removed_from_cache', False)}")
    return 0


def _cmd_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule list",
        description="List capsules currently stored in the local capsule store.",
    )
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    rows = list_capsules(Path(args.capsules_dir) if args.capsules_dir else None)
    if bool(args.json):
        _print_json(
            {
                "schema_version": CAPSULES_JSON_SCHEMA,
                "capsules": [_capsule_json_entry(row) for row in rows],
            }
        )
        return 0

    if not rows:
        print("(capsule store is empty)")
        return 0

    print("stored capsules:")
    for row in rows:
        aliases = _render_aliases(row)
        print(
            f"- {row.get('capsule_id')} | aliases: {aliases} | path: {row.get('path')}"
        )
    return 0


def _cmd_show(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule show",
        description="Show one stored capsule entry from the local capsule store.",
        epilog="Examples:\n  lelabo capsule show my_capsule\n  lelabo capsule show my_alias --json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("id_or_alias", help="Stored capsule id or alias to inspect")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        row = get_capsule(args.id_or_alias, Path(args.capsules_dir) if args.capsules_dir else None)
        if row is None:
            raise ValueError(f"Unknown capsule '{args.id_or_alias}'")
    except ValueError as exc:
        raise SystemExit(str(exc))
    if bool(args.json):
        _print_json(_capsule_command_json("show", row))
    else:
        _print_entry_block("stored capsule:", row)
    return 0


def _cmd_remove(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule remove",
        description="Remove one stored capsule entry from the local capsule store.",
        epilog="Examples:\n  lelabo capsule remove my_capsule\n  lelabo capsule remove my_capsule --keep-files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("id_or_alias", help="Stored capsule id or alias to remove")
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
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
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

    if bool(args.json):
        _print_json(
            _capsule_command_json(
                "remove",
                removed,
                result_fields=("deleted_files", "delete_files_requested", "allow_external_delete"),
            )
        )
    else:
        _print_action_block(
            "removed capsule from store:",
            removed,
            extra_fields=("deleted_files", "delete_files_requested", "allow_external_delete"),
        )
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(CAPSULE_HELP)
        return 0

    cmd = args[0]
    rest = args[1:]

    if cmd == "init":
        return _cmd_init(rest)
    if cmd == "pack":
        return _cmd_pack(rest)
    if cmd == "install":
        return _cmd_install(rest)
    if cmd == "stash":
        return _cmd_stash(rest)
    if cmd == "checkout":
        return _cmd_checkout(rest)
    if cmd == "list":
        return _cmd_list(rest)
    if cmd == "show":
        return _cmd_show(rest)
    if cmd == "remove":
        return _cmd_remove(rest)

    raise SystemExit(
        f"Unknown capsule subcommand: {cmd}\n\n"
        "Use one of: init, stash, checkout, install, pack, list, show, remove.\n"
        "Run `lelabo capsule -h` for usage."
    )
