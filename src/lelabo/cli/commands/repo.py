"""Publish repo management flows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from ...capsule import get_capsule, inspect_capsule_directory
from ...capsule.discovery import DiscoveredCapsule, discover_workspace_capsules, find_capsule_root, is_capsule_root
from ...capsule.publish import (
    create_repo,
    current_github_login,
    list_configured_targets,
    load_publish_state,
    remove_configured_target,
    save_configured_target,
    set_default_target,
)
from ...config.user_settings import load_effective_settings
from ..interactive_picker import pick_many_with_checkboxes
from ..ui import print_block, print_list_block, print_status


REPO_JSON_SCHEMA = "lelabo.cli.repo/v1"
REPOS_JSON_SCHEMA = "lelabo.cli.repos/v1"
_REPO_ACTION_EXISTING = "__existing_repo__"
_REPO_ACTION_CREATE = "__create_repo__"
_REPO_ACTION_CANCEL = "__cancel__"
_REPO_KIND_SHARED = "__shared_repo__"
_REPO_KIND_DEDICATED = "__dedicated_repo__"


REPO_HELP = """\
Manage publish repos and targets for capsules.

Usage:
  lelabo repo <subcommand> [args]

Subcommands:
  list         List configured GitHub repos
  create       Create a shared GitHub repo
  add capsule  Add one capsule to an existing or new GitHub repo
  edit         Edit repo attachments and defaults
  detach       Detach one capsule from a configured GitHub repo
"""


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


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


def _is_interactive_tty() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _default_owner(settings: dict[str, Any]) -> str:
    github_cfg = settings.get("github", {}) if isinstance(settings.get("github", {}), dict) else {}
    return str(github_cfg.get("owner") or current_github_login() or "").strip()


def _default_visibility(settings: dict[str, Any], override: str | None = None) -> str:
    github_cfg = settings.get("github", {}) if isinstance(settings.get("github", {}), dict) else {}
    return str(override or github_cfg.get("default_visibility") or "private").strip().lower() or "private"


def _default_branch(settings: dict[str, Any]) -> str:
    github_cfg = settings.get("github", {}) if isinstance(settings.get("github", {}), dict) else {}
    return str(github_cfg.get("default_branch") or "main").strip() or "main"


def _confirm(question: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return bool(default)
    return answer in {"y", "yes"}


def _prompt_with_default(label: str, default: str) -> str:
    token = str(default).strip()
    prompt = f"{label} [{token}]: " if token else f"{label}: "
    answer = input(prompt).strip()
    return answer or token


def _pick_one(title: str, text: str, options: Sequence[tuple[str, str]], *, cancel_message: str) -> str:
    selected = pick_many_with_checkboxes(
        title=title,
        text=text,
        options=options,
        selection_noun="option",
        confirm_button_text="Select current option",
        max_selection_count=1,
        max_selection_message="Select exactly one option before confirming.",
    )
    if selected is None:
        raise SystemExit(cancel_message)
    token = str(selected[0]).strip() if selected else ""
    if not token:
        raise SystemExit(cancel_message)
    return token


def _pick_capsule_root(candidates: Sequence[DiscoveredCapsule], *, title: str, cancel_message: str) -> Path:
    token = _pick_one(
        title,
        "Select one capsule.",
        [(item.path, f"{item.capsule_id} | path: {item.path}") for item in candidates],
        cancel_message=cancel_message,
    )
    for item in candidates:
        if item.path == token:
            return item.root
    raise SystemExit(cancel_message)


def _resolve_capsule_root(capsule_ref: str | None, *, caps_dir: Path | None, purpose: str) -> Path:
    if not capsule_ref:
        candidates = list(discover_workspace_capsules(start=Path.cwd()))
        if len(candidates) == 1:
            return candidates[0].root
        if not candidates:
            raise SystemExit("No capsule found in the current workspace. Pass a capsule path/id or create a capsule under this directory.")
        if _is_interactive_tty():
            return _pick_capsule_root(candidates, title="Select capsule", cancel_message=f"{purpose} canceled by user.")
        names = ", ".join(item.capsule_id for item in candidates)
        raise SystemExit(f"Multiple capsules found in the current workspace: {names}. Pass a capsule path/id.")

    ref_path = Path(capsule_ref).expanduser()
    if ref_path.exists():
        start = ref_path.resolve()
        if start.is_file():
            start = start.parent
        root = find_capsule_root(start=start)
        if root is None and is_capsule_root(start):
            root = start
        if root is None:
            raise SystemExit(f"Path '{capsule_ref}' is not inside a capsule (missing capsule.toml).")
        return root.resolve()

    local_candidate = (Path.cwd() / str(capsule_ref)).resolve()
    if is_capsule_root(local_candidate):
        return local_candidate

    row = get_capsule(capsule_ref, caps_dir)
    if row is None:
        raise SystemExit(f"Unknown capsule '{capsule_ref}' (not found as path nor stored id/alias).")
    root = Path(str(row.get("path", ""))).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Capsule path does not exist on disk: {root}")
    return root


def _parse_repo_full_name(token: str) -> tuple[str, str]:
    owner, _, repo = str(token).strip().partition("/")
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        raise SystemExit("GitHub repo must be formatted as owner/repo.")
    return owner, repo


def _list_github_repos(owner: str) -> list[str]:
    owner_token = str(owner).strip()
    if not owner_token:
        raise SystemExit("GitHub owner is required. Set --owner or `lelabo config set github.owner <owner>`.")
    proc = subprocess.run(
        ["gh", "repo", "list", owner_token, "--limit", "1000", "--json", "nameWithOwner"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or "Run `gh auth login` before listing GitHub repos."
        raise SystemExit(err)
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit("Failed to parse GitHub repo list.") from exc
    repos: list[str] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        full_name = str(item.get("nameWithOwner", "")).strip()
        if full_name:
            repos.append(full_name)
    return sorted(set(repos))


def _pick_existing_repo_full_name(owner: str) -> str:
    repos = _list_github_repos(owner)
    if not repos:
        raise SystemExit(f"No GitHub repos found for {owner}. Create a repo first.")
    options = [(item, item) for item in repos] + [(_REPO_ACTION_CANCEL, "Cancel")]
    selected = _pick_one(
        "Select GitHub repo",
        "Select one existing GitHub repo.",
        options,
        cancel_message="Repo selection canceled by user.",
    )
    if selected == _REPO_ACTION_CANCEL:
        raise SystemExit("Repo selection canceled by user.")
    return selected


def _prompt_repo_kind() -> str:
    selected = _pick_one(
        "Select new repo type",
        "Choose how this capsule should be published.",
        (
            (_REPO_KIND_SHARED, "Shared repo"),
            (_REPO_KIND_DEDICATED, "Dedicated repo for this capsule"),
            (_REPO_ACTION_CANCEL, "Cancel"),
        ),
        cancel_message="Repo creation canceled by user.",
    )
    if selected == _REPO_ACTION_CANCEL:
        raise SystemExit("Repo creation canceled by user.")
    return selected


def _repo_label(owner: str, repo: str) -> str:
    return f"{owner}/{repo}"


def _capsule_id_for_path(capsule_root: Path) -> str:
    root = capsule_root.expanduser().resolve()
    if root.exists() and root.is_dir():
        try:
            return str(inspect_capsule_directory(root).get("capsule_id", root.name)).strip() or root.name
        except Exception:
            return root.name
    return root.name


def _configured_target_rows(*, capsule_root: Path | None = None) -> list[dict[str, Any]]:
    if capsule_root is not None:
        root = capsule_root.expanduser().resolve()
        rows: list[dict[str, Any]] = []
        for item in list_configured_targets(root):
            row = dict(item)
            row["capsule_path"] = str(root)
            row["capsule_id"] = _capsule_id_for_path(root)
            rows.append(row)
        return rows

    state = load_publish_state()
    rows = []
    for capsule_path, entry in sorted((state.get("capsules") or {}).items()):
        if not isinstance(entry, dict):
            continue
        targets = entry.get("targets", {})
        if not isinstance(targets, dict):
            continue
        root = Path(str(capsule_path)).expanduser().resolve()
        capsule_id = _capsule_id_for_path(root)
        for name, target in sorted(targets.items()):
            if not isinstance(target, dict):
                continue
            row = dict(target)
            row["name"] = str(name)
            row["capsule_path"] = str(root)
            row["capsule_id"] = capsule_id
            row["default"] = str(entry.get("default_target") or "") == str(name)
            row["last_used"] = str(entry.get("last_used_target") or "") == str(name)
            rows.append(row)
    rows.sort(
        key=lambda item: (
            str(item.get("owner", "")),
            str(item.get("repo", "")),
            str(item.get("capsule_id", "")),
            str(item.get("path", "")),
            str(item.get("name", "")),
        )
    )
    return rows


def _group_repo_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        owner = str(row.get("owner", "")).strip()
        repo = str(row.get("repo", "")).strip()
        if not owner or not repo:
            continue
        key = (owner, repo)
        group = grouped.setdefault(
            key,
            {
                "owner": owner,
                "repo": repo,
                "capsules": [],
                "attachments": [],
            },
        )
        capsule_id = str(row.get("capsule_id", "")).strip()
        if capsule_id and capsule_id not in group["capsules"]:
            group["capsules"].append(capsule_id)
        group["attachments"].append(dict(row))
    out = list(grouped.values())
    for group in out:
        group["capsules"] = sorted(group["capsules"])
        group["attachments"] = sorted(
            group["attachments"],
            key=lambda item: (
                str(item.get("capsule_id", "")),
                str(item.get("path", "")),
                str(item.get("name", "")),
            ),
        )
    out.sort(key=lambda item: (str(item.get("owner", "")), str(item.get("repo", ""))))
    return out


def _repo_group_label(group: dict[str, Any]) -> str:
    capsules = ", ".join(str(item) for item in list(group.get("capsules", [])) if str(item).strip()) or "-"
    return f"{group.get('owner', '-')}/{group.get('repo', '-')} | capsules: {capsules}"


def _pick_repo_group(groups: Sequence[dict[str, Any]], *, title: str, cancel_message: str) -> dict[str, Any]:
    if not groups:
        raise SystemExit("No configured GitHub repos were found.")
    token = _pick_one(
        title,
        "Select one GitHub repo.",
        [
            (_repo_label(str(group.get("owner", "")), str(group.get("repo", ""))), _repo_group_label(group))
            for group in groups
        ],
        cancel_message=cancel_message,
    )
    for group in groups:
        if _repo_label(str(group.get("owner", "")), str(group.get("repo", ""))) == token:
            return group
    raise SystemExit(cancel_message)


def _attachment_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('capsule_id', '-')} | "
        f"folder: {row.get('path', '-')} | "
        f"default: {bool(row.get('default'))}"
    )


def _pick_attachment(rows: Sequence[dict[str, Any]], *, title: str, cancel_message: str) -> dict[str, Any]:
    if not rows:
        raise SystemExit("No repo attachment was found.")
    if len(rows) == 1:
        return dict(rows[0])
    token = _pick_one(
        title,
        "Select one capsule attachment.",
        [
            (
                f"{row.get('capsule_path')}::{row.get('name')}",
                _attachment_label(row),
            )
            for row in rows
        ],
        cancel_message=cancel_message,
    )
    capsule_path, _, name = token.partition("::")
    for row in rows:
        if str(row.get("capsule_path")) == capsule_path and str(row.get("name")) == name:
            return dict(row)
    raise SystemExit(cancel_message)


def _resolve_repo_group(repo_ref: str | None, capsule_ref: str | None, *, caps_dir: Path | None, purpose: str) -> dict[str, Any]:
    rows = _configured_target_rows(
        capsule_root=_resolve_capsule_root(capsule_ref, caps_dir=caps_dir, purpose=purpose) if capsule_ref else None
    )
    groups = _group_repo_rows(rows)
    token = str(repo_ref or "").strip()
    if token:
        owner, repo = _parse_repo_full_name(token)
        matches = [group for group in groups if str(group.get("owner")) == owner and str(group.get("repo")) == repo]
        if len(matches) == 1:
            return matches[0]
        raise SystemExit(f"Unknown configured GitHub repo '{owner}/{repo}'.")
    if _is_interactive_tty():
        return _pick_repo_group(groups, title="Select GitHub repo", cancel_message=f"{purpose} canceled by user.")
    raise SystemExit("No repo was specified. Run interactively to pick one, or pass owner/repo.")


def _suggest_target_name(existing_targets: Sequence[dict[str, Any]], *, owner: str, repo: str) -> str:
    normalized_owner = str(owner).strip()
    normalized_repo = str(repo).strip()
    for item in existing_targets:
        if (
            str(item.get("kind", "")).strip() == "github"
            and str(item.get("owner", "")).strip() == normalized_owner
            and str(item.get("repo", "")).strip() == normalized_repo
        ):
            name = str(item.get("name", "")).strip()
            if name:
                return name
    used = {str(item.get("name", "")).strip() for item in existing_targets if str(item.get("name", "")).strip()}
    if "github" not in used:
        return "github"
    repo_token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in normalized_repo).strip("-_") or "repo"
    candidate = f"github-{repo_token}"
    if candidate not in used:
        return candidate
    index = 2
    while True:
        numbered = f"{candidate}-{index}"
        if numbered not in used:
            return numbered
        index += 1


def _build_target_payload(
    *,
    capsule_root: Path,
    existing_targets: Sequence[dict[str, Any]],
    owner: str,
    repo: str,
    folder: str,
    visibility: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    folder_token = str(folder).strip() or inspect_capsule_directory(capsule_root).get("capsule_id") or capsule_root.name
    if folder_token == "":
        raise SystemExit("Folder cannot be empty.")
    visibility_token = str(visibility).strip().lower() or "private"
    if visibility_token not in {"public", "private"}:
        raise SystemExit("Visibility must be 'public' or 'private'.")
    return {
        "name": _suggest_target_name(existing_targets, owner=owner, repo=repo),
        "kind": "github",
        "owner": owner,
        "repo": repo,
        "branch": _default_branch(settings),
        "path": folder_token,
        "visibility": visibility_token,
    }


def create_shared_repo_interactively(
    *,
    settings: dict[str, Any],
    owner_override: str | None = None,
    repo_override: str | None = None,
    visibility_override: str | None = None,
    create_remote_repo: bool = True,
    show_summary: bool = True,
) -> dict[str, Any]:
    owner = _prompt_with_default("Owner", str(owner_override or _default_owner(settings)).strip())
    if not owner:
        raise SystemExit("GitHub owner is required.")
    repo = _prompt_with_default("Repo", str(repo_override or "lelabo-capsules").strip() or "lelabo-capsules")
    visibility = _prompt_with_default("Visibility", _default_visibility(settings, visibility_override))
    visibility = str(visibility).strip().lower() or "private"
    if visibility not in {"public", "private"}:
        raise SystemExit("Visibility must be 'public' or 'private'.")
    if show_summary:
        print_block(
            "Repo",
            (
                ("owner", owner),
                ("repo", repo),
                ("visibility", visibility),
            ),
        )
    question = "Create repo?" if create_remote_repo else "Use this repo in preview?"
    if not _confirm(question, default=True):
        raise SystemExit("Repo creation canceled by user.")
    if create_remote_repo:
        create_repo(owner, repo, visibility)
    return {"owner": owner, "repo": repo, "visibility": visibility}


def configure_github_target_for_capsule(
    capsule_root: Path,
    *,
    settings: dict[str, Any],
    persist: bool,
    owner_override: str | None = None,
    repo_override: str | None = None,
    visibility_override: str | None = None,
    make_default_default: bool = True,
    create_remote_repo: bool | None = None,
    show_summary: bool = True,
) -> dict[str, Any]:
    info = inspect_capsule_directory(capsule_root)
    capsule_id = str(info.get("capsule_id", capsule_root.name)).strip() or capsule_root.name
    default_owner = str(owner_override or _default_owner(settings)).strip()
    default_visibility = _default_visibility(settings, visibility_override)
    existing_targets = [item for item in list_configured_targets(capsule_root) if str(item.get("kind", "")).strip() == "github"]

    selected_owner = str(owner_override or "").strip()
    selected_repo = str(repo_override or "").strip()
    selected_visibility = default_visibility
    default_folder = capsule_id

    if selected_owner and selected_repo:
        pass
    else:
        action = _pick_one(
            "Select repo action",
            "Choose how to attach this capsule.",
            (
                (_REPO_ACTION_EXISTING, "Existing GitHub repo"),
                (_REPO_ACTION_CREATE, "Create new GitHub repo"),
                (_REPO_ACTION_CANCEL, "Cancel"),
            ),
            cancel_message="Repo target setup canceled by user.",
        )
        if action == _REPO_ACTION_CANCEL:
            raise SystemExit("Repo target setup canceled by user.")
        if action == _REPO_ACTION_EXISTING:
            owner = _prompt_with_default("Owner", default_owner)
            selected_owner, selected_repo = _parse_repo_full_name(_pick_existing_repo_full_name(owner))
        else:
            repo_kind = _prompt_repo_kind()
            owner = _prompt_with_default("Owner", default_owner)
            if not owner:
                raise SystemExit("GitHub owner is required.")
            repo_default = "lelabo-capsules" if repo_kind == _REPO_KIND_SHARED else capsule_id
            folder_default = capsule_id if repo_kind == _REPO_KIND_SHARED else "."
            repo = _prompt_with_default("Repo", repo_default)
            visibility = _prompt_with_default("Visibility", default_visibility)
            visibility = str(visibility).strip().lower() or "private"
            if visibility not in {"public", "private"}:
                raise SystemExit("Visibility must be 'public' or 'private'.")
            if show_summary:
                print_block(
                    "Repo",
                    (
                        ("owner", owner),
                        ("repo", repo),
                        ("visibility", visibility),
                    ),
                )
            question = "Create repo?" if create_remote_repo is not False else "Use this repo in preview?"
            if not _confirm(question, default=True):
                raise SystemExit("Repo creation canceled by user.")
            if create_remote_repo is not False:
                create_repo(owner, repo, visibility)
            selected_owner = owner
            selected_repo = repo
            selected_visibility = visibility
            default_folder = folder_default

    if not selected_owner or not selected_repo:
        raise SystemExit("GitHub repo selection is incomplete.")

    folder = _prompt_with_default("Folder", default_folder)
    visibility = _prompt_with_default("Visibility", selected_visibility)
    set_default = _confirm("Set as default target?", default=make_default_default)
    target = _build_target_payload(
        capsule_root=capsule_root,
        existing_targets=existing_targets,
        owner=selected_owner,
        repo=selected_repo,
        folder=folder,
        visibility=visibility,
        settings=settings,
    )
    if show_summary:
        print_block(
            "Publish target",
            (
                ("capsule", capsule_id),
                ("repo", _repo_label(selected_owner, selected_repo)),
                ("folder", target.get("path")),
                ("visibility", target.get("visibility")),
                ("default", set_default),
            ),
        )
    question = "Save target?" if persist else "Use this target in preview?"
    if not _confirm(question, default=True):
        raise SystemExit("Repo target setup canceled by user.")
    if persist:
        return save_configured_target(capsule_root, target, make_default=set_default)
    ephemeral = dict(target)
    ephemeral["default"] = False
    ephemeral["last_used"] = False
    ephemeral["_ephemeral"] = True
    return ephemeral


def _cmd_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo list",
        description="List configured GitHub repos.",
    )
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="Repo listing") if args.capsule_ref else None
    groups = _group_repo_rows(_configured_target_rows(capsule_root=capsule_root))
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPOS_JSON_SCHEMA,
                "command": "list",
                "capsule_path": str(capsule_root) if capsule_root is not None else None,
                "repos": groups,
            }
        )
    else:
        if not groups:
            print_status("info", "No configured GitHub repos were found.")
            return 0
        print_list_block(
            "GitHub repos",
            [
                f"{group.get('owner', '-')}/{group.get('repo', '-')} | capsules: {', '.join(group.get('capsules', []))}"
                for group in groups
            ],
        )
    return 0


def _cmd_create(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo create",
        description="Create one shared GitHub repo.",
    )
    parser.add_argument("--owner", default=None, help="GitHub owner")
    parser.add_argument("--repo", default=None, help="GitHub repository name")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="Create a public GitHub repo")
    vis.add_argument("--private", action="store_true", help="Create a private GitHub repo")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    visibility = "public" if bool(args.public) else "private" if bool(args.private) else None
    repo_info = create_shared_repo_interactively(
        settings=settings,
        owner_override=args.owner,
        repo_override=args.repo,
        visibility_override=visibility,
        create_remote_repo=True,
        show_summary=not bool(args.json),
    )
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPO_JSON_SCHEMA,
                "command": "create",
                "repo": repo_info,
            }
        )
    else:
        print_status("success", "GitHub repo created.")
        print_block("Repo", (("owner", repo_info["owner"]), ("repo", repo_info["repo"]), ("visibility", repo_info["visibility"])))
    return 0


def _cmd_add_capsule(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo add capsule",
        description="Add one capsule to an existing or new GitHub repo.",
    )
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--owner", default=None, help="GitHub owner")
    parser.add_argument("--repo", default=None, help="GitHub repository name")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="Use public visibility for a new repo or target")
    vis.add_argument("--private", action="store_true", help="Use private visibility for a new repo or target")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="Repo target setup")
    visibility = "public" if bool(args.public) else "private" if bool(args.private) else None
    target = configure_github_target_for_capsule(
        capsule_root,
        settings=settings,
        persist=True,
        owner_override=args.owner,
        repo_override=args.repo,
        visibility_override=visibility,
        make_default_default=True,
        create_remote_repo=True,
        show_summary=not bool(args.json),
    )
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPO_JSON_SCHEMA,
                "command": "add-capsule",
                "capsule_path": str(capsule_root),
                "target": target,
            }
        )
    else:
        print_status("success", "Publish target added.")
        print_block(
            "Target",
            (
                ("name", target.get("name")),
                ("kind", target.get("kind")),
                ("owner", target.get("owner")),
                ("repo", target.get("repo")),
                ("branch", target.get("branch")),
                ("folder", target.get("path")),
                ("default", target.get("default")),
            ),
        )
    return 0


def _cmd_edit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo edit",
        description="Edit repo attachments and defaults.",
    )
    parser.add_argument("repo_ref", nargs="?", default=None, help="Optional configured GitHub repo as owner/repo")
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias to filter repos")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    group = _resolve_repo_group(args.repo_ref, args.capsule_ref, caps_dir=caps_dir, purpose="Repo edit")
    action = _pick_one(
        "Edit repo",
        "Choose one action for this repo.",
        (
            ("default", "Set default repo for a capsule"),
            ("detach", "Detach capsule"),
            (_REPO_ACTION_CANCEL, "Cancel"),
        ),
        cancel_message="Repo edit canceled by user.",
    )
    if action == _REPO_ACTION_CANCEL:
        raise SystemExit("Repo edit canceled by user.")
    selected = _pick_attachment(
        list(group.get("attachments", [])),
        title="Select capsule",
        cancel_message="Repo edit canceled by user.",
    )
    capsule_root = Path(str(selected.get("capsule_path", ""))).expanduser().resolve()
    if action == "default":
        try:
            target = set_default_target(capsule_root, str(selected.get("name", "")))
        except ValueError as exc:
            raise SystemExit(str(exc))
        command_name = "edit-default"
        success_message = "Default repo updated."
    else:
        try:
            target = remove_configured_target(capsule_root, str(selected.get("name", "")))
        except ValueError as exc:
            raise SystemExit(str(exc))
        command_name = "edit-detach"
        success_message = "Capsule detached from repo."
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPO_JSON_SCHEMA,
                "command": command_name,
                "capsule_path": str(capsule_root),
                "target": target,
            }
        )
    else:
        print_status("success", success_message)
        print_block(
            "Target",
            (
                ("capsule", selected.get("capsule_id")),
                ("repo", _repo_label(str(selected.get("owner", "")), str(selected.get("repo", "")))),
                ("folder", selected.get("path")),
                ("default", target.get("default")),
            ),
        )
    return 0


def _cmd_detach(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo repo detach",
        description="Detach one capsule from a configured GitHub repo.",
    )
    parser.add_argument("repo_ref", nargs="?", default=None, help="Optional configured GitHub repo as owner/repo")
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    group = _resolve_repo_group(args.repo_ref, args.capsule_ref, caps_dir=caps_dir, purpose="Repo detach")
    attachments = list(group.get("attachments", []))
    if args.capsule_ref:
        capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="Repo detach")
        attachments = [row for row in attachments if Path(str(row.get("capsule_path", ""))).expanduser().resolve() == capsule_root]
        if not attachments:
            raise SystemExit(
                f"Capsule '{_capsule_id_for_path(capsule_root)}' is not attached to {_repo_label(str(group.get('owner', '')), str(group.get('repo', '')))}."
            )
    selected = _pick_attachment(attachments, title="Select capsule", cancel_message="Repo detach canceled by user.")
    repo_name = _repo_label(str(selected.get("owner", "")), str(selected.get("repo", "")))
    if not _confirm(f"Detach this capsule from {repo_name}?", default=False):
        raise SystemExit("Repo detach canceled by user.")
    capsule_root = Path(str(selected.get("capsule_path", ""))).expanduser().resolve()
    try:
        target = remove_configured_target(capsule_root, str(selected.get("name", "")))
    except ValueError as exc:
        raise SystemExit(str(exc))
    if bool(args.json):
        _print_json(
            {
                "schema_version": REPO_JSON_SCHEMA,
                "command": "detach",
                "capsule_path": str(capsule_root),
                "target": target,
            }
        )
    else:
        print_status("success", "Capsule detached from repo.")
        print_block(
            "Target",
            (
                ("capsule", selected.get("capsule_id")),
                ("repo", repo_name),
                ("folder", selected.get("path")),
            ),
        )
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(REPO_HELP)
        return 0
    cmd = str(args[0]).strip().lower()
    rest = args[1:]
    if cmd == "list":
        return _cmd_list(rest)
    if cmd == "create":
        return _cmd_create(rest)
    if cmd == "add":
        if rest and str(rest[0]).strip().lower() == "capsule":
            return _cmd_add_capsule(rest[1:])
        return _cmd_add_capsule(rest)
    if cmd == "edit":
        return _cmd_edit(rest)
    if cmd == "detach":
        return _cmd_detach(rest)
    raise SystemExit(
        f"Unknown repo subcommand: {cmd}\n\n"
        "Use one of: list, create, add capsule, edit, detach.\n"
        "Run `lelabo repo -h` for usage."
    )


__all__ = [
    "configure_github_target_for_capsule",
    "create_shared_repo_interactively",
    "main",
]
