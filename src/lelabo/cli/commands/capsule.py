"""CLI entrypoint for capsule lifecycle and store operations."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ...capsule import (
    attach_capsule,
    checkout_capsule,
    inspect_capsule_directory,
    create_capsule_scaffold,
    get_capsule,
    install_capsule,
    install_capsule_from_directory,
    list_capsules,
    remove_capsule,
    stash_capsule,
)
from ...capsule.discovery import (
    DiscoveredCapsule,
    discover_visible_capsules,
    discover_workspace_capsules,
    resolve_visible_capsule_ref,
)
from ...capsule.github import clone_github_repo, is_github_repo_url
from ...capsule.plugins.discovery import find_active_capsule_root
from ..interactive_picker import pick_many_with_checkboxes
from ..ui import print_block, print_list_block, print_status
from ...config.user_settings import load_effective_settings


CAPSULE_HELP = """\
Manage LeLabo experiment capsules.

Usage:
  lelabo capsule <subcommand> [args]

Subcommands:
  init       Create a local work capsule in the current workspace
  attach     Link an external capsule into the LeLabo capsule registry (no move/copy)
  stash      Move a local capsule into the local capsule store/cache
  checkout   Move a stored capsule back into a local workspace
  install    Import an external capsule bundle or GitHub repo into the local store
  list       List visible capsules from the workspace and store
  show       Show one visible capsule entry
  remove     Remove one stored capsule entry (and files by default)

Help:
  lelabo capsule -h
  lelabo capsule <subcommand> -h
"""

CAPSULE_JSON_SCHEMA = "lelabo.cli.capsule/v1"
CAPSULES_JSON_SCHEMA = "lelabo.cli.capsules/v1"
CAPSULE_LIST_JSON_SCHEMA = "lelabo.cli.capsules/v2"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _print_json(payload: Any) -> None:
    print(_json_dumps(payload))


def _render_aliases(row: dict[str, Any]) -> str:
    aliases = [str(item).strip() for item in list(row.get("aliases", []) or []) if str(item).strip()]
    return ", ".join(aliases) if aliases else "-"


def _capsule_json_entry(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "capsule_id": row.get("capsule_id"),
        "aliases": list(row.get("aliases", []) or []),
        "path": row.get("path"),
    }
    if "kind" in row:
        payload["kind"] = row.get("kind")
    if "status" in row:
        payload["status"] = row.get("status")
    return payload


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
    rows: list[tuple[str, Any]] = [
        ("capsule_id", row.get("capsule_id", "-")),
        ("aliases", _render_aliases(row)),
    ]
    if str(row.get("status", "")).strip():
        rows.append(("status", row.get("status")))
    if str(row.get("path", "")).strip():
        rows.append(("path", row.get("path")))
    print_block(title, rows)


def _print_action_block(title: str, row: dict[str, Any], *, extra_fields: Sequence[str]) -> None:
    rows: list[tuple[str, Any]] = [
        ("capsule_id", row.get("capsule_id", "-")),
        ("aliases", _render_aliases(row)),
    ]
    if str(row.get("status", "")).strip():
        rows.append(("status", row.get("status")))
    if str(row.get("path", "")).strip():
        rows.append(("path", row.get("path")))
    for field in extra_fields:
        if field in row:
            rows.append((field, row[field]))
    print_block(title, rows)


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


def _is_interactive_tty() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _select_repo_capsules(
    repo_root: Path,
    repo_capsules: Sequence[DiscoveredCapsule],
    *,
    requested_capsules: Sequence[str] | None,
    install_all: bool,
) -> list[dict[str, str]]:
    capsules = list(repo_capsules)
    if not capsules:
        raise SystemExit("This GitHub repo does not contain any LeLabo capsules.")
    if install_all:
        return [{"id": item.capsule_id, "path": item.path} for item in capsules]
    requested_tokens = [str(item).strip() for item in list(requested_capsules or []) if str(item).strip()]
    if requested_tokens:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for token in requested_tokens:
            matches = [item for item in capsules if item.capsule_id == token]
            if not matches:
                available = ", ".join(sorted(item.capsule_id for item in capsules))
                raise SystemExit(f"Unknown repo capsule '{token}'. Available: {available}")
            if len(matches) > 1:
                paths = ", ".join(item.path for item in matches)
                raise SystemExit(f"Repo capsule id '{token}' is ambiguous across: {paths}")
            resolved = matches[0]
            if resolved.path in seen:
                continue
            seen.add(resolved.path)
            out.append({"id": resolved.capsule_id, "path": resolved.path})
        return out
    if len(capsules) == 1:
        item = capsules[0]
        return [{"id": item.capsule_id, "path": item.path}]
    if not _is_interactive_tty():
        raise SystemExit(
            "This GitHub repo contains multiple capsules. "
            "Use `--capsule <id>` (repeatable) or `--all`."
        )

    selected_ids = pick_many_with_checkboxes(
        title="Select capsules to install",
        text="Use Space to toggle capsules, then press Enter to confirm.",
        options=[
            (item.root.relative_to(repo_root).as_posix(), f"{item.capsule_id} | path: {item.root.relative_to(repo_root).as_posix()}")
            for item in capsules
        ],
    )
    if selected_ids is None:
        raise SystemExit("Install canceled by user.")
    if not selected_ids:
        raise SystemExit("Install canceled: no capsule selected.")

    out: list[dict[str, str]] = []
    for token in selected_ids:
        for item in capsules:
            if item.root.relative_to(repo_root).as_posix() == str(token).strip():
                out.append({"id": item.capsule_id, "path": item.path})
                break
    if not out:
        raise SystemExit("Install canceled: no valid capsule selection.")
    return out


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

    print_status("success", "Capsule initialized.")
    print_block("Capsule", (("name", args.name), ("path", out)))
    return 0


def _cmd_install(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule install",
        description="Import an external capsule bundle or GitHub repo into the local capsule store.",
        epilog=(
            "Examples:\n"
            "  lelabo capsule install demo_capsule.tar.gz\n"
            "  lelabo capsule install https://github.com/owner/repo --capsule demo\n"
            "  lelabo capsule install https://github.com/owner/repo --all --ref main"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", help="Capsule bundle path or GitHub LeLabo repo URL")
    parser.add_argument("--alias", default=None, help="Optional alias inside the capsule store")
    parser.add_argument(
        "--rename-to",
        default=None,
        help="Install under a different capsule id to avoid id conflicts in store.",
    )
    parser.add_argument(
        "--force-replace",
        action="store_true",
        help="Replace an existing stored capsule with the same capsule id.",
    )
    parser.add_argument(
        "--capsule",
        action="append",
        default=[],
        help="Capsule id to install from a GitHub repo (repeatable).",
    )
    parser.add_argument("--all", action="store_true", help="Install all capsules found in a GitHub repo")
    parser.add_argument("--ref", default=None, help="Optional git ref (branch/tag/commit) for GitHub installs")
    parser.add_argument(
        "--checkout",
        nargs="?",
        const="__DEFAULT__",
        default=None,
        metavar="DEST",
        help="After install, checkout into DEST. If omitted, uses config default checkout directory.",
    )
    parser.add_argument("--capsules-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    source = str(args.source).strip()
    if not source:
        raise SystemExit("Install source cannot be empty.")
    selected_capsules_cli = [str(item).strip() for item in list(args.capsule or []) if str(item).strip()]
    if bool(selected_capsules_cli) and bool(args.all):
        raise SystemExit("`--capsule` and `--all` cannot be used together.")
    if bool(args.all) and args.rename_to is not None:
        raise SystemExit("`--rename-to` cannot be used with `lelabo capsule install --all`.")
    if bool(args.all) and args.alias is not None:
        raise SystemExit("`--alias` cannot be used with `lelabo capsule install --all`.")

    try:
        if is_github_repo_url(source):
            with tempfile.TemporaryDirectory(prefix="lelabo_capsule_git_install_") as td:
                if not bool(args.json):
                    print_status("info", "Cloning LeLabo repo...")
                clone_info = clone_github_repo(
                    repo_url=source,
                    destination=Path(td) / "repo",
                    ref=args.ref,
                )
                repo_root = Path(td) / "repo"
                repo_capsules = list(discover_workspace_capsules(start=repo_root))
                if not bool(args.json):
                    print_status("info", "Resolving capsules...")
                selected_capsules = _select_repo_capsules(
                    repo_root,
                    repo_capsules,
                    requested_capsules=selected_capsules_cli,
                    install_all=bool(args.all),
                )
                if not bool(args.json):
                    print_status(
                        "info",
                        "Installing selected capsules..."
                        if len(selected_capsules) > 1
                        else "Installing selected capsule...",
                    )
                installed_rows: list[dict[str, Any]] = []
                for selected in selected_capsules:
                    source_dir = Path(selected["path"]).resolve()
                    display_path = source_dir.relative_to(repo_root).as_posix()
                    source_meta = {
                        "type": "github_repo",
                        "path": clone_info["repo_url"],
                        "requested_ref": clone_info["requested_ref"],
                        "ref": clone_info["resolved_ref"],
                        "capsule_id": selected["id"],
                        "capsule_path": display_path,
                    }
                    incoming_info = inspect_capsule_directory(
                        source_dir,
                        source_meta=source_meta,
                        capsule_id_override=args.rename_to if len(selected_capsules) == 1 else None,
                    )
                    active_root = find_active_capsule_root()
                    workspace_candidate = Path.cwd() / str(incoming_info.get("capsule_id", ""))
                    if active_root is None and _is_capsule_root(workspace_candidate):
                        active_root = workspace_candidate.resolve()
                    if active_root is not None:
                        active_info = inspect_capsule_directory(active_root)
                        if (
                            str(incoming_info.get("capsule_id", "")) == str(active_info.get("capsule_id", ""))
                            and str(incoming_info.get("fingerprint", "")) == str(active_info.get("fingerprint", ""))
                        ):
                            entry = {
                                "capsule_id": str(incoming_info.get("capsule_id", "")),
                                "aliases": [],
                                "path": str(active_root.resolve()),
                                "kind": incoming_info.get("kind"),
                                "source_kind": "github_repo",
                                "source_url": clone_info["repo_url"],
                                "source_repo": f"{clone_info['owner']}/{clone_info['repo']}",
                                "source_capsule": selected["id"],
                                "requested_ref": clone_info["requested_ref"],
                                "source_ref": clone_info["resolved_ref"],
                                "install_action": "already_present_workspace",
                                "replaced_existing": False,
                                "checked_out": False,
                            }
                            installed_rows.append(entry)
                            continue
                    entry = install_capsule_from_directory(
                        source_dir=source_dir,
                        alias=args.alias if len(selected_capsules) == 1 else None,
                        capsules_dir=caps_dir,
                        source_bundle=(
                            f"github_repo:{clone_info['repo_url']}@{clone_info['resolved_ref']}#{display_path}"
                        ),
                        source_meta=source_meta,
                        capsule_id_override=args.rename_to if len(selected_capsules) == 1 else None,
                        force_replace=bool(args.force_replace),
                    )
                    entry["source_kind"] = "github_repo"
                    entry["source_url"] = clone_info["repo_url"]
                    entry["source_repo"] = f"{clone_info['owner']}/{clone_info['repo']}"
                    entry["source_capsule"] = selected["id"]
                    entry["source_ref"] = clone_info["resolved_ref"]
                    entry["requested_ref"] = clone_info["requested_ref"]
                    installed_rows.append(entry)
                entries = installed_rows
        else:
            if _looks_like_remote_source(source):
                raise ValueError(
                    f"Unsupported remote source '{source}'. "
                    "V1 install supports GitHub repo URLs only."
                )
            local_source = Path(source).expanduser().resolve()
            if local_source.exists() and local_source.is_dir():
                raise ValueError(
                    f"Install source '{local_source}' is a local capsule directory. "
                    "Use `lelabo capsule attach <capsule_dir>` to link local capsules."
                )
            else:
                if not bool(args.json):
                    print_status("info", "Installing capsule bundle...")
                entries = [install_capsule(
                    bundle_path=local_source,
                    alias=args.alias,
                    capsules_dir=caps_dir,
                    capsule_id_override=args.rename_to,
                    force_replace=bool(args.force_replace),
                )]
                entries[0]["source_kind"] = "bundle"
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))

    checkout_target: Path | None = None
    if args.checkout is not None:
        checkout_target = _default_checkout_dir(settings) if args.checkout == "__DEFAULT__" else Path(args.checkout).expanduser().resolve()
    elif _install_checkout_enabled(settings):
        checkout_target = _default_checkout_dir(settings)

    for entry in entries:
        if checkout_target is not None and entry.get("install_action") != "already_present_workspace":
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

    if len(entries) == 1 and not bool(args.all):
        entry = entries[0]
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
                        "source_repo",
                        "source_capsule",
                        "requested_ref",
                        "source_ref",
                        "install_action",
                        "replaced_existing",
                        "checked_out",
                        "checked_out_path",
                        "removed_from_cache",
                    ),
                )
            )
        else:
            title = "Capsule install"
            if entry.get("install_action") == "already_present_workspace":
                print_status("info", "This capsule is already present in the current workspace.")
            else:
                print_status("success", "Capsule install complete.")
            _print_action_block(
                title,
                entry,
                extra_fields=(
                    "installed_at",
                    "source_bundle",
                    "source_kind",
                    "source_url",
                    "source_repo",
                    "source_capsule",
                    "requested_ref",
                    "source_ref",
                    "install_action",
                    "replaced_existing",
                    "checked_out",
                    "checked_out_path",
                ),
            )
        return 0

    payload = {
        "schema_version": CAPSULES_JSON_SCHEMA,
        "command": "install",
        "count": len(entries),
        "capsules": [
            {
                **_capsule_json_entry(row),
                "result": {
                    field: row[field]
                    for field in (
                        "installed_at",
                        "source_bundle",
                        "source_kind",
                        "source_url",
                        "source_repo",
                        "source_capsule",
                        "requested_ref",
                        "source_ref",
                        "install_action",
                        "replaced_existing",
                        "checked_out",
                        "checked_out_path",
                    )
                    if field in row
                },
            }
            for row in entries
        ],
    }
    if bool(args.json):
        _print_json(payload)
    else:
        print_status("success", f"Installed {len(entries)} capsules from the repo.")
        print_list_block(
            "Install results",
            [
                f"{row.get('capsule_id')} | action: {row.get('install_action')} | path: {row.get('path')}"
                for row in entries
            ],
        )
    return 0


def _cmd_attach(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule attach",
        description="Link an external capsule into the LeLabo capsule registry without moving files.",
        epilog=(
            "Examples:\n"
            "  lelabo capsule attach /external/my_capsule\n"
            "  lelabo capsule attach /external/my_capsule --alias paper_demo\n"
            "  lelabo capsule attach /external/my_capsule --rename-to paper_v2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", nargs="?", default=None, help="Local capsule root (defaults to active capsule from cwd)")
    parser.add_argument("--alias", default=None, help="Optional alias in the capsule registry")
    parser.add_argument("--rename-to", default=None, help="Register under a different capsule id")
    parser.add_argument(
        "--force-replace",
        action="store_true",
        help="Replace existing registry entry when capsule id already exists",
    )
    parser.add_argument("--capsules-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    source = Path(args.source).expanduser() if args.source else None
    try:
        entry = attach_capsule(
            alias=args.alias,
            source_path=source,
            capsules_dir=caps_dir,
            capsule_id_override=args.rename_to,
            force_replace=bool(args.force_replace),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))

    if bool(args.json):
        _print_json(
            _capsule_command_json(
                "attach",
                entry,
                result_fields=("attached_from", "moved", "attach_action", "replaced_existing"),
            )
        )
    else:
        _print_action_block(
            "attached capsule:",
            entry,
            extra_fields=("attached_from", "moved", "attach_action", "replaced_existing"),
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
    parser.add_argument("--capsules-dir", default=None, help=argparse.SUPPRESS)
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
                print_status("success", f"Stashed {len(stored)} capsules.")
                print_list_block(
                    "Stored capsules",
                    [f"{row.get('capsule_id')} -> {row.get('path')}" for row in stored],
                )
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
    parser.add_argument("--capsules-dir", default=None, help=argparse.SUPPRESS)
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
        print_status("success", "Capsule checked out.")
        print_block(
            "Checkout",
            (
                ("capsule_id", restored.get("capsule_id", "-")),
                ("source_path", restored.get("source_path", "-")),
                ("checked_out_path", restored.get("checked_out_path", "-")),
                ("removed_from_cache", restored.get("removed_from_cache", False)),
            ),
        )
    return 0


def _cmd_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule list",
        description="List visible capsules from the workspace and local store.",
    )
    parser.add_argument("--capsules-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    rows = [
        {
            "capsule_id": item.capsule_id,
            "aliases": list(item.aliases),
            "path": item.path,
            "status": item.status,
        }
        for item in discover_visible_capsules(start=Path.cwd(), capsules_dir=caps_dir)
    ]
    if bool(args.json):
        _print_json(
            {
                "schema_version": CAPSULE_LIST_JSON_SCHEMA,
                "capsules": [_capsule_json_entry(row) for row in rows],
            }
        )
        return 0

    if not rows:
        print_status("info", "No capsules found in the current workspace or store.")
        return 0

    print_list_block(
        "Capsules",
        [
            f"{row.get('capsule_id')} | status: {row.get('status')} | aliases: {_render_aliases(row)} | path: {row.get('path')}"
            for row in rows
        ],
    )
    return 0


def _cmd_show(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule show",
        description="Show one visible capsule from the workspace or local store.",
        epilog="Examples:\n  lelabo capsule show my_capsule\n  lelabo capsule show my_alias --json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("id_or_alias", help="Visible capsule id or alias to inspect")
    parser.add_argument("--capsules-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)

    try:
        visible = resolve_visible_capsule_ref(
            args.id_or_alias,
            start=Path.cwd(),
            capsules_dir=caps_dir,
            command="lelabo capsule show",
            usage="lelabo capsule show <capsule>",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    row = {
        "capsule_id": visible.capsule_id,
        "aliases": list(visible.aliases),
        "path": visible.path,
        "status": visible.status,
    }
    for stored in list_capsules(caps_dir):
        raw_path = str(stored.get("path", "")).strip()
        if not raw_path:
            continue
        try:
            stored_root = Path(raw_path).expanduser().resolve()
        except OSError:
            continue
        if stored_root != visible.root:
            continue
        for key, value in stored.items():
            if key in {"capsule_id", "aliases", "path"}:
                continue
            row.setdefault(key, value)
        break
    if visible.status == "workspace":
        info = inspect_capsule_directory(visible.root)
        kind = str(info.get("kind", "")).strip()
        if kind:
            row.setdefault("kind", kind)
    if bool(args.json):
        _print_json(_capsule_command_json("show", row))
    else:
        _print_entry_block("Visible capsule", row)
    return 0


def _cmd_remove(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule remove",
        description="Remove one stored capsule entry from the local capsule store.",
        epilog="Examples:\n  lelabo capsule remove my_capsule\n  lelabo capsule remove my_capsule --keep-files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("id_or_alias", help="Stored capsule id or alias to remove")
    parser.add_argument("--capsules-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Only remove from capsule registry, keep capsule files on disk",
    )
    parser.add_argument(
        "-r",
        action="store_true",
        dest="rm_recursive",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-f",
        action="store_true",
        dest="rm_force",
        help=argparse.SUPPRESS,
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
            "Removed capsule",
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
    if cmd == "install":
        return _cmd_install(rest)
    if cmd == "attach":
        return _cmd_attach(rest)
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
        "Use one of: init, attach, stash, checkout, install, list, show, remove.\n"
        "Run `lelabo capsule -h` for usage."
    )
