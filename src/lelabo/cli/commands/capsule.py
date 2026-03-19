"""CLI entrypoint for capsule lifecycle and store operations."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ...capsule import (
    attach_capsule,
    add_capsule_to_gitspace,
    checkout_capsule,
    inspect_capsule_directory,
    create_capsule_scaffold,
    current_github_login,
    find_gitspace_root,
    git_repo_root,
    get_capsule,
    init_gitspace,
    install_capsule,
    install_capsule_from_directory,
    list_capsules,
    load_gitspace,
    pack_capsule,
    remove_capsule,
    resolve_gitspace_capsule,
    share_gitspace_github,
    stash_capsule,
)
from ...capsule.github import clone_github_repo, is_github_repo_url
from ...capsule.plugins.discovery import find_active_capsule_root
from ...capsule.share import parse_owner_repo_from_origin
from ..interactive_picker import pick_many_with_checkboxes
from .push import run_push_command
from ..ui import print_block, print_list_block, print_status
from ...config.user_settings import load_effective_settings


CAPSULE_HELP = """\
Manage LeLabo experiment capsules.

Usage:
  lelabo capsule <subcommand> [args]

Subcommands:
  init       Create a local work capsule in the current workspace
  attach     Link a local capsule into the LeLabo capsule registry (no move/copy)
  stash      Move a local capsule into the local capsule store/cache
  checkout   Move a stored capsule back into a local workspace
  install    Import an external capsule bundle or GitHub multi-capsule repo into the local store
  share      Transition alias for `lelabo push` (or export a local bundle)
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
    rows: list[tuple[str, Any]] = [
        ("capsule_id", row.get("capsule_id", "-")),
        ("aliases", _render_aliases(row)),
    ]
    if str(row.get("path", "")).strip():
        rows.append(("path", row.get("path")))
    print_block(title, rows)


def _print_action_block(title: str, row: dict[str, Any], *, extra_fields: Sequence[str]) -> None:
    rows: list[tuple[str, Any]] = [
        ("capsule_id", row.get("capsule_id", "-")),
        ("aliases", _render_aliases(row)),
    ]
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


def _confirm(question: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return bool(default)
    return answer in {"y", "yes"}


def _prompt_with_default(label: str, default: str) -> str:
    token = str(default).strip()
    answer = input(f"{label} [{token}]: ").strip()
    return answer or token


def _select_gitspace_capsules(
    gitspace: dict[str, Any],
    *,
    requested_capsules: Sequence[str] | None,
    install_all: bool,
) -> list[dict[str, str]]:
    capsules = list(gitspace.get("capsules", []) or [])
    if not capsules:
        raise SystemExit("This LeLabo repo does not declare any capsules.")
    if install_all:
        return [{"id": str(item["id"]), "path": str(item["path"])} for item in capsules]
    requested_tokens = [str(item).strip() for item in list(requested_capsules or []) if str(item).strip()]
    if requested_tokens:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for token in requested_tokens:
            try:
                resolved = resolve_gitspace_capsule(gitspace, token)
            except ValueError as exc:
                raise SystemExit(str(exc))
            capsule_id = str(resolved["id"])
            if capsule_id in seen:
                continue
            seen.add(capsule_id)
            out.append(resolved)
        return out
    if len(capsules) == 1:
        item = capsules[0]
        return [{"id": str(item["id"]), "path": str(item["path"])}]
    if not _is_interactive_tty():
        raise SystemExit(
            "This LeLabo repo contains multiple capsules. "
            "Use `--capsule <id>` (repeatable) or `--all`."
        )

    selected_ids = pick_many_with_checkboxes(
        title="Select capsules to install",
        text=(
            f"Repo manifest: {gitspace.get('name', '-')}\n"
            "Use Space to toggle capsules, then press Enter to confirm."
        ),
        options=[(str(item["id"]), f"{item['id']}  |  path: {item['path']}") for item in capsules],
    )
    if selected_ids is None:
        raise SystemExit("Install canceled by user.")
    if not selected_ids:
        raise SystemExit("Install canceled: no capsule selected.")

    out: list[dict[str, str]] = []
    for token in selected_ids:
        try:
            out.append(resolve_gitspace_capsule(gitspace, token))
        except ValueError:
            continue
    if not out:
        raise SystemExit("Install canceled: no valid capsule selection.")
    return out


def _default_gitspace_root_for_capsule(capsule_root: Path) -> Path:
    try:
        repo_root = git_repo_root(capsule_root)
    except Exception:
        return capsule_root.parent.resolve()
    return repo_root.resolve()


def _bootstrap_gitspace_for_share(
    *,
    capsule_root: Path,
    owner: str | None,
    repo: str | None,
    branch: str | None,
    visibility: str,
    create_repo_if_missing: bool,
    default_branch: str,
    assume_yes: bool,
    emit_messages: bool,
) -> tuple[Path, str, str, str, dict[str, Any]]:
    gitspace_root = _default_gitspace_root_for_capsule(capsule_root)
    detected_login = current_github_login()
    default_name = str(gitspace_root.name or capsule_root.parent.name or capsule_root.name).strip() or "gitspace"
    resolved_owner = str(owner or "").strip() or str(detected_login or "").strip()
    resolved_repo = str(repo or "").strip() or default_name
    resolved_visibility = str(visibility).strip().lower() or "private"

    if not assume_yes:
        if not _is_interactive_tty():
            raise SystemExit(
                "This capsule does not belong to any gitspace. "
                "Run interactively to bootstrap one, or pass `--yes` with enough GitHub defaults configured."
            )
        if emit_messages:
            print_status("info", "This capsule does not belong to any gitspace yet.")
            print_block(
                "Bootstrap target",
                (
                    ("capsule_root", capsule_root),
                    ("suggested_gitspace_root", gitspace_root),
                    ("github_account", detected_login or "-"),
                ),
            )
        if not _confirm("Create a gitspace for this capsule now?", default=True):
            raise SystemExit("Gitspace bootstrap canceled by user.")
        default_name = _prompt_with_default("Gitspace name", default_name)
        resolved_owner = _prompt_with_default("GitHub owner", resolved_owner or (detected_login or ""))
        resolved_repo = _prompt_with_default("GitHub repo", resolved_repo or default_name)
        resolved_visibility = _prompt_with_default("Visibility (private/public)", resolved_visibility)
        if emit_messages:
            print_block(
                "Review",
                (
                    ("gitspace_root", gitspace_root),
                    ("gitspace_name", default_name),
                    ("owner", resolved_owner),
                    ("repo", resolved_repo),
                    ("visibility", resolved_visibility),
                ),
            )
        if not _confirm("Continue?", default=True):
            raise SystemExit("Gitspace bootstrap canceled by user.")
    else:
        if not resolved_owner:
            raise SystemExit(
                "This capsule does not belong to any gitspace and GitHub owner could not be resolved. "
                "Set --owner or `lelabo config set github.owner <owner>`."
            )

    if resolved_visibility not in {"public", "private"}:
        raise SystemExit("Visibility must be 'public' or 'private'.")

    if emit_messages:
        print_status("info", "Creating gitspace manifest and publishing to GitHub...")
    gitspace = init_gitspace(gitspace_root, name=default_name)
    gitspace = add_capsule_to_gitspace(capsule_root, gitspace_root=gitspace_root)
    result = share_gitspace_github(
        gitspace_root=gitspace_root,
        owner=resolved_owner,
        repo=resolved_repo,
        branch=branch,
        visibility=resolved_visibility,
        create_repo_if_missing=create_repo_if_missing,
        default_branch=default_branch,
        auto_init_git=True,
        assume_yes=True,
    )
    result["bootstrapped_gitspace"] = True
    result["gitspace_name"] = gitspace["name"]
    result["shared_capsule_id"] = gitspace["capsule"]["id"]
    return gitspace_root, resolved_owner, resolved_repo, resolved_visibility, result


def _resolve_share_capsule_root(capsule_ref: str | None, *, caps_dir: Path | None) -> Path:
    if not capsule_ref:
        active = find_active_capsule_root()
        if active is None:
            raise SystemExit("No active capsule found. Pass a capsule path/id or run inside a capsule directory.")
        return active.resolve()

    ref_path = Path(capsule_ref).expanduser()
    if ref_path.exists():
        start = ref_path.resolve()
        if start.is_file():
            start = start.parent
        root = find_active_capsule_root(start=start)
        if root is None:
            raise SystemExit(f"Path '{capsule_ref}' is not inside a capsule (missing capsule.toml).")
        return root.resolve()

    row = get_capsule(capsule_ref, caps_dir)
    if row is None:
        raise SystemExit(f"Unknown capsule '{capsule_ref}' (not found as path nor stored id/alias).")
    root = Path(str(row.get("path", ""))).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Capsule path does not exist on disk: {root}")
    return root


def _export_capsule_local_bundle(capsule_root: Path, *, out_path: Path | None) -> Path:
    root = capsule_root.resolve()
    if out_path is None:
        out = (Path.cwd() / f"{root.name}.tar.gz").resolve()
    else:
        out = out_path.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, mode="w:gz") as tf:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if ".git" in rel.parts:
                continue
            if "__pycache__" in rel.parts:
                continue
            if p.suffix in {".pyc", ".pyo"}:
                continue
            tf.add(p, arcname=f"{root.name}/{rel.as_posix()}")
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

    print_status("success", "Capsule scaffold created.")
    print_block("Capsule", (("path", out),))
    print("Next: open README.md to start.")
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
    print_status("success", "Capsule bundle created.")
    print_block("Bundle", (("path", out),))
    return 0


def _cmd_install(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule install",
        description="Import an external capsule bundle or GitHub multi-capsule repo into the local capsule store.",
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
        help="Capsule id to install from a GitHub multi-capsule repo (repeatable).",
    )
    parser.add_argument("--all", action="store_true", help="Install all capsules declared by a GitHub LeLabo repo")
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
                try:
                    gitspace = load_gitspace(repo_root)
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"GitHub repo '{clone_info['repo_url']}' is not a LeLabo multi-capsule repo. "
                        "Add `.lelabo/gitspace.toml` and declare at least one capsule."
                    ) from exc
                if not bool(args.json):
                    print_status("info", "Resolving capsules...")
                selected_capsules = _select_gitspace_capsules(
                    gitspace,
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
                    source_dir = repo_root / selected["path"]
                    source_meta = {
                        "type": "github_gitspace",
                        "path": clone_info["repo_url"],
                        "requested_ref": clone_info["requested_ref"],
                        "ref": clone_info["resolved_ref"],
                        "gitspace": {
                            "name": gitspace["name"],
                            "capsule_id": selected["id"],
                            "capsule_path": selected["path"],
                        },
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
                                "source_kind": "github_gitspace",
                                "source_url": clone_info["repo_url"],
                                "source_repo": f"{clone_info['owner']}/{clone_info['repo']}",
                                "source_gitspace": gitspace["name"],
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
                            f"github_gitspace:{clone_info['repo_url']}@{clone_info['resolved_ref']}#{selected['id']}"
                        ),
                        source_meta=source_meta,
                        capsule_id_override=args.rename_to if len(selected_capsules) == 1 else None,
                        force_replace=bool(args.force_replace),
                    )
                    entry["source_kind"] = "github_gitspace"
                    entry["source_url"] = clone_info["repo_url"]
                    entry["source_repo"] = f"{clone_info['owner']}/{clone_info['repo']}"
                    entry["source_gitspace"] = gitspace["name"]
                    entry["source_capsule"] = selected["id"]
                    entry["source_ref"] = clone_info["resolved_ref"]
                    entry["requested_ref"] = clone_info["requested_ref"]
                    installed_rows.append(entry)
                entries = installed_rows
        else:
            if _looks_like_remote_source(source):
                raise ValueError(
                    f"Unsupported remote source '{source}'. "
                    "V1 install supports GitHub LeLabo repo URLs only."
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
                        "source_gitspace",
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
                    "source_gitspace",
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
                        "source_gitspace",
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
        description="Link a local capsule into the LeLabo capsule registry without moving files.",
        epilog=(
            "Examples:\n"
            "  lelabo capsule attach ./my_capsule\n"
            "  lelabo capsule attach --alias paper_demo\n"
            "  lelabo capsule attach ./my_capsule --rename-to paper_v2"
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
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path")
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
        print_status("info", "The capsule store is empty.")
        return 0

    print_list_block(
        "Stored capsules",
        [
            f"{row.get('capsule_id')} | aliases: {_render_aliases(row)} | path: {row.get('path')}"
            for row in rows
        ],
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
        _print_entry_block("Stored capsule", row)
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
            "Removed capsule",
            removed,
            extra_fields=("deleted_files", "delete_files_requested", "allow_external_delete"),
        )
    return 0


def _cmd_share(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo capsule share",
        description="Publish a capsule with `lelabo push`, or export a local bundle.",
        epilog=(
            "Examples:\n"
            "  lelabo capsule share\n"
            "  lelabo capsule share my_capsule_alias --owner owner --repo repo\n"
            "  lelabo capsule share --mode local --out ./my_capsule.tar.gz"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--mode", choices=["github", "local"], default="github", help="Share mode backend")
    parser.add_argument("--owner", default=None, help="GitHub owner override")
    parser.add_argument("--repo", default=None, help="GitHub repository override")
    parser.add_argument("--branch", default=None, help="Target git branch override")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="Create/share as a public repo")
    vis.add_argument("--private", action="store_true", help="Create/share as a private repo")
    parser.add_argument("--out", default=None, help="Output bundle path for --mode local")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive confirmation for gitspace bootstrap and automatic git initialization.",
    )
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path (for capsule id/alias refs)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_share_capsule_root(args.capsule_ref, caps_dir=caps_dir)

    if args.mode == "local":
        out = _export_capsule_local_bundle(
            capsule_root,
            out_path=Path(args.out).expanduser() if args.out else None,
        )
        payload = {
            "schema_version": CAPSULE_JSON_SCHEMA,
            "command": "share",
            "target": "local",
            "result": {
                "capsule_path": str(capsule_root),
                "bundle_path": str(out),
                "mode": "local",
            },
        }
        if bool(args.json):
            _print_json(payload)
        else:
            print_status("success", "Capsule exported locally.")
            print_block(
                "Local share",
                (
                    ("capsule_path", capsule_root),
                    ("bundle_path", out),
                ),
            )
        return 0

    forward_argv: list[str] = []
    if args.capsule_ref is not None:
        forward_argv.append(str(args.capsule_ref))
    if args.owner:
        forward_argv.extend(["--owner", str(args.owner)])
    if args.repo:
        forward_argv.extend(["--repo", str(args.repo)])
    if args.branch:
        forward_argv.extend(["--branch", str(args.branch)])
    if bool(args.public):
        forward_argv.append("--public")
    if bool(args.private):
        forward_argv.append("--private")
    if bool(args.yes):
        forward_argv.append("--yes")
    if args.capsules_dir:
        forward_argv.extend(["--capsules-dir", str(args.capsules_dir)])
    if bool(args.json):
        forward_argv.append("--json")

    try:
        _, push_payload = run_push_command(forward_argv, prog="lelabo capsule share")
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    if bool(args.json):
        result = push_payload.get("result")
        if result is None:
            results = list(push_payload.get("results", []) or [])
            result = results[0] if results else {}
        _print_json(
            {
                "schema_version": CAPSULE_JSON_SCHEMA,
                "command": "share",
                "target": result.get("target_kind", "github"),
                "result": result,
            }
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
    if cmd == "attach":
        return _cmd_attach(rest)
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
        "Use one of: init, attach, stash, checkout, install, share, pack, list, show, remove.\n"
        "Run `lelabo capsule -h` for usage."
    )
