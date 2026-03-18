"""CLI entrypoint for capsule lifecycle and store operations."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ...capsule import (
    checkout_capsule,
    create_capsule_scaffold,
    get_capsule,
    install_capsule,
    install_capsule_from_directory,
    list_capsules,
    pack_capsule,
    remove_capsule,
    share_capsule_github,
    stash_capsule,
)
from ...capsule.github import clone_github_repo, is_github_repo_url, parse_owner_repo_spec
from ...capsule.plugins.discovery import find_active_capsule_root
from ...capsule.share import parse_owner_repo_from_origin
from ...config.user_settings import load_effective_settings


CAPSULE_HELP = """\
Manage LeLabo experiment capsules.

Usage:
  lelabo capsule <subcommand> [args]

Subcommands:
  init       Create a local work capsule in the current workspace
  sweep      List or run sweep configs from the active capsule
  stash      Move a local capsule into the local capsule store/cache
  checkout   Move a stored capsule back into a local workspace
  install    Import an external capsule bundle or GitHub repo into the local store
  share      Share the active capsule workspace to GitHub
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


def _default_checkout_dir(settings: dict[str, Any]) -> Path:
    cfg = settings.get("capsules", {})
    if not isinstance(cfg, dict):
        return Path(".").resolve()
    raw = str(cfg.get("default_checkout_dir", ".") or ".").strip() or "."
    return Path(raw).expanduser().resolve()


def _install_checkout_enabled(settings: dict[str, Any]) -> bool:
    cfg = settings.get("capsules", {})
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("install_checkout", False))


def _looks_like_remote_source(raw: str) -> bool:
    token = str(raw).strip()
    return "://" in token or token.startswith("git@")


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
        description="Import an external capsule bundle or GitHub repo into the local capsule store.",
        epilog=(
            "Examples:\n"
            "  lelabo capsule install demo_capsule.tar.gz\n"
            "  lelabo capsule install https://github.com/owner/repo --alias demo\n"
            "  lelabo capsule install https://github.com/owner/repo --ref v1.0.0 --checkout ."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="Capsule bundle path or GitHub repo URL")
    parser.add_argument("--alias", default=None, help="Optional alias inside the capsule store")
    parser.add_argument("--ref", default=None, help="Optional git ref (branch/tag/commit) for GitHub installs")
    parser.add_argument(
        "--checkout",
        nargs="?",
        const="__DEFAULT__",
        default=None,
        metavar="DEST",
        help="After install, checkout into DEST. If omitted, uses config default checkout directory.",
    )
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    source = str(args.source).strip()
    if not source:
        raise SystemExit("Install source cannot be empty.")

    try:
        if is_github_repo_url(source):
            with tempfile.TemporaryDirectory(prefix="lelabo_capsule_git_install_") as td:
                clone_info = clone_github_repo(
                    repo_url=source,
                    destination=Path(td) / "repo",
                    ref=args.ref,
                )
                source_meta = {
                    "type": "github",
                    "path": clone_info["repo_url"],
                    "requested_ref": clone_info["requested_ref"],
                    "ref": clone_info["resolved_ref"],
                }
                entry = install_capsule_from_directory(
                    source_dir=Path(td) / "repo",
                    alias=args.alias,
                    capsules_dir=caps_dir,
                    source_bundle=f"github:{clone_info['repo_url']}@{clone_info['resolved_ref']}",
                    source_meta=source_meta,
                )
                entry["source_kind"] = "github"
                entry["source_url"] = clone_info["repo_url"]
                entry["source_ref"] = clone_info["resolved_ref"]
                entry["requested_ref"] = clone_info["requested_ref"]
        else:
            if _looks_like_remote_source(source):
                raise ValueError(
                    f"Unsupported remote source '{source}'. "
                    "V1 install supports GitHub repo URLs only."
                )
            entry = install_capsule(
                bundle_path=Path(source),
                alias=args.alias,
                capsules_dir=caps_dir,
            )
            entry["source_kind"] = "bundle"
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))

    checkout_target: Path | None = None
    if args.checkout is not None:
        checkout_target = _default_checkout_dir(settings) if args.checkout == "__DEFAULT__" else Path(args.checkout).expanduser().resolve()
    elif _install_checkout_enabled(settings):
        checkout_target = _default_checkout_dir(settings)

    if checkout_target is not None:
        checkout_result = checkout_capsule(
            capsule_or_alias=args.alias or str(entry.get("capsule_id", "")),
            destination_dir=checkout_target,
            capsules_dir=caps_dir,
        )
        entry["checked_out"] = True
        entry["checked_out_path"] = checkout_result.get("checked_out_path")
        entry["removed_from_cache"] = checkout_result.get("removed_from_cache")
    else:
        entry["checked_out"] = False

    if bool(args.json):
        _print_json(
            _capsule_command_json(
                "install",
                entry,
                result_fields=(
                    "installed_at",
                    "source_bundle",
                    "source_kind",
                    "source_url",
                    "requested_ref",
                    "source_ref",
                    "checked_out",
                    "checked_out_path",
                    "removed_from_cache",
                ),
            )
        )
    else:
        _print_action_block(
            "installed capsule into store:",
            entry,
            extra_fields=(
                "installed_at",
                "source_bundle",
                "source_kind",
                "source_url",
                "requested_ref",
                "source_ref",
                "checked_out",
                "checked_out_path",
            ),
        )
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

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
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
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)

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

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    rows = list_capsules(caps_dir)
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
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)

    try:
        row = get_capsule(args.id_or_alias, caps_dir)
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
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    allow_external_delete = bool(args.force_external_delete or (args.rm_recursive and args.rm_force))
    if (args.rm_recursive or args.rm_force) and not allow_external_delete:
        raise SystemExit("Use '-rf' together to allow external capsule deletion.")

    try:
        removed = remove_capsule(
            capsule_or_alias=args.id_or_alias,
            capsules_dir=caps_dir,
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


def _cmd_share(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule share",
        description="Share the active capsule workspace to GitHub.",
        epilog=(
            "Examples:\n"
            "  lelabo capsule share github\n"
            "  lelabo capsule share github owner/repo\n"
            "  lelabo capsule share github --owner owner --repo repo --public"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("target", choices=["github"], help="Share target backend")
    parser.add_argument("repo_spec", nargs="?", default=None, help="Optional owner/repo target")
    parser.add_argument("--owner", default=None, help="GitHub owner override")
    parser.add_argument("--repo", default=None, help="GitHub repository override")
    parser.add_argument("--branch", default=None, help="Target branch override")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="Create/share as a public repo")
    vis.add_argument("--private", action="store_true", help="Create/share as a private repo")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    if args.target != "github":
        raise SystemExit(f"Unsupported share target '{args.target}'.")

    capsule_root = find_active_capsule_root()
    if capsule_root is None:
        raise SystemExit("No active capsule found. Run share from inside a capsule directory.")

    settings = _effective_settings()
    github_cfg = settings.get("github", {}) if isinstance(settings.get("github", {}), dict) else {}

    spec_owner = spec_repo = None
    if args.repo_spec:
        try:
            spec_owner, spec_repo = parse_owner_repo_spec(args.repo_spec)
        except ValueError as exc:
            raise SystemExit(str(exc))

    origin_spec = parse_owner_repo_from_origin(capsule_root)
    origin_owner, origin_repo = origin_spec if origin_spec is not None else (None, None)

    owner = str(args.owner or spec_owner or origin_owner or github_cfg.get("owner", "")).strip()
    if not owner:
        raise SystemExit("GitHub owner is required. Set --owner or `lelabo config set github.owner <owner>`.")
    default_repo = origin_repo or capsule_root.name
    repo = str(args.repo or spec_repo or default_repo).strip()
    if not repo:
        raise SystemExit("GitHub repo is required.")

    if bool(args.public):
        visibility = "public"
    elif bool(args.private):
        visibility = "private"
    else:
        visibility = str(github_cfg.get("default_visibility", "private")).strip().lower() or "private"
    if visibility not in {"public", "private"}:
        raise SystemExit("Visibility must be 'public' or 'private'.")

    branch = str(args.branch or github_cfg.get("default_branch", "main")).strip() or "main"
    create_repo_if_missing = bool(github_cfg.get("create_repo_if_missing", True))

    try:
        result = share_capsule_github(
            capsule_root=capsule_root,
            owner=owner,
            repo=repo,
            branch=branch,
            visibility=visibility,
            create_repo_if_missing=create_repo_if_missing,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    if bool(args.json):
        payload = {
            "schema_version": CAPSULE_JSON_SCHEMA,
            "command": "share",
            "target": "github",
            "result": result,
        }
        _print_json(payload)
    else:
        print("shared capsule to github:")
        print(f"owner: {result.get('owner')}")
        print(f"repo: {result.get('repo')}")
        print(f"branch: {result.get('branch')}")
        print(f"remote_url: {result.get('remote_url')}")
        print(f"created_repo: {result.get('created_repo')}")
        print(f"created_origin_remote: {result.get('created_origin_remote')}")
        print(f"pushed: {result.get('pushed')}")
    return 0


def _cmd_sweep(argv: list[str]) -> int:
    """List or run sweep configs from the active capsule."""
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print("Usage:")
        print("  lelabo capsule sweep              List available sweep configs")
        print("  lelabo capsule sweep run <name>    Run a sweep from the active capsule")
        print()
        return 0

    capsule_root = find_active_capsule_root()
    if capsule_root is None:
        raise SystemExit("No active capsule found. Run from inside a capsule directory.")

    if argv[0] == "run":
        if len(argv) < 2:
            raise SystemExit("Usage: lelabo capsule sweep run <name> [--dry-run] [--max-parallel N] [--gpus G]")

        sweep_name = argv[1]
        candidates = [
            capsule_root / "sweeps" / f"{sweep_name}.yaml",
            capsule_root / "sweeps" / f"{sweep_name}.yml",
            capsule_root / "configs" / f"{sweep_name}.yaml",
            capsule_root / "configs" / f"{sweep_name}.yml",
        ]
        config_path = next((p for p in candidates if p.exists()), None)
        if config_path is None:
            raise SystemExit(
                f"Sweep config '{sweep_name}' not found in capsule. "
                f"Searched: {[str(p) for p in candidates]}"
            )

        from .sweep import _cmd_run
        return _cmd_run(["--config", str(config_path)] + argv[2:])

    # Default: list available sweep configs
    sweep_dirs = [capsule_root / "sweeps", capsule_root / "configs"]
    found: list[tuple[str, str]] = []
    for d in sweep_dirs:
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix in {".yaml", ".yml"} and f.is_file():
                    found.append((f.stem, str(f.relative_to(capsule_root))))

    if not found:
        print("No sweep configs found in active capsule.")
        return 0

    print("Available sweep configs:")
    for name, rel_path in found:
        print(f"  {name}  ({rel_path})")
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(CAPSULE_HELP)
        return 0

    cmd = args[0]
    rest = args[1:]

    if cmd == "sweep":
        return _cmd_sweep(rest)
    if cmd == "init":
        return _cmd_init(rest)
    if cmd == "pack":
        return _cmd_pack(rest)
    if cmd == "install":
        return _cmd_install(rest)
    if cmd == "share":
        return _cmd_share(rest)
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
        "Use one of: init, sweep, stash, checkout, install, share, pack, list, show, remove.\n"
        "Run `lelabo capsule -h` for usage."
    )
