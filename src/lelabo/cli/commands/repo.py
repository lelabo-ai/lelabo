"""Publish target management flows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from ...capsule import inspect_capsule_directory, install_capsule_from_directory
from ...capsule.discovery import (
    DiscoveredCapsule,
    resolve_visible_capsule_ref,
)
from ...capsule.publish import (
    create_repo,
    current_github_login,
    github_repo_metadata,
    get_global_target,
    inspect_target_repo_capsules,
    list_configured_targets,
    list_global_targets,
    load_publish_state,
    remove_configured_target,
    remove_global_target,
    remove_target_repo_capsule,
    repo_exists,
    save_configured_target,
    save_global_target,
    set_default_target,
    unset_default_target,
)
from ...config.user_settings import load_effective_settings
from ..interactive_picker import pick_many_with_checkboxes
from ..ui import print_block, print_status


TARGET_JSON_SCHEMA = "lelabo.cli.target/v1"
TARGETS_JSON_SCHEMA = "lelabo.cli.targets/v1"
_REPO_ACTION_EXISTING = "__existing_repo__"
_REPO_ACTION_EXISTING_REMOTE = "__existing_remote_repo__"
_REPO_ACTION_CREATE = "__create_repo__"
_REPO_ACTION_CANCEL = "__cancel__"


REPO_HELP = """\
Manage GitHub publish targets for capsules.

Usage:
  lelabo targets <subcommand> [args]

Subcommands:
  list         List configured GitHub targets
  create       Create or register one shared GitHub target
  attach       Attach one capsule to a configured GitHub target
  detach       Detach one capsule from a configured GitHub target
  defaults     Manage default targets for one capsule
  import       Import capsule(s) from a target repo
  remove-capsule Remove one capsule from a target repo and detach it locally
  delete       Delete one local target entry and clean its attachments
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
    normalized = str(purpose).strip().lower()
    command = "lelabo targets edit"
    usage = "lelabo targets edit <owner/repo> <capsule>"
    if normalized == "target attach":
        command = "lelabo targets attach"
        usage = "lelabo targets attach <target> <capsule>"
    if normalized == "target detach":
        command = "lelabo targets detach"
        usage = "lelabo targets detach <target> <capsule>"
    elif normalized == "target listing":
        command = "lelabo targets list"
        usage = "lelabo targets list <capsule>"
    elif normalized == "target defaults":
        command = "lelabo targets defaults"
        usage = "lelabo targets defaults <subcommand> ..."
    elif normalized == "target import":
        command = "lelabo targets import"
        usage = "lelabo targets import <target>"
    elif normalized == "target remove-capsule":
        command = "lelabo targets remove-capsule"
        usage = "lelabo targets remove-capsule <target> <capsule>"
    try:
        return resolve_visible_capsule_ref(
            capsule_ref,
            start=Path.cwd(),
            capsules_dir=caps_dir,
            command=command,
            usage=usage,
        ).root
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_repo_full_name(token: str) -> tuple[str, str]:
    owner, _, repo = str(token).strip().partition("/")
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        raise SystemExit("GitHub target must be formatted as owner/repo.")
    return owner, repo


def _list_github_repos(owner: str) -> list[str]:
    owner_token = str(owner).strip()
    if not owner_token:
        raise SystemExit("GitHub owner is required. Set `github.owner` or pass owner/repo explicitly.")
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
        token = str(item.get("nameWithOwner", "")).strip()
        if token:
            repos.append(token)
    return sorted(set(repos))


def _prompt_owner_repo(settings: dict[str, Any], owner_repo_override: str | None = None) -> tuple[str, str]:
    owner_default = _default_owner(settings)
    combined_default = str(owner_repo_override or "").strip()
    if not combined_default and owner_default:
        combined_default = f"{owner_default}/lelabo-capsules"
    token = _prompt_with_default("Target", combined_default)
    owner, repo = _parse_repo_full_name(token)
    return owner, repo


def _pick_existing_target(global_targets: Sequence[dict[str, Any]], *, cancel_message: str) -> dict[str, Any]:
    if not global_targets:
        raise SystemExit("No configured GitHub targets were found. Create one first.")
    token = _pick_one(
        "Select GitHub target",
        "Select one configured GitHub target.",
        [
            (
                _repo_label(str(item.get("owner", "")), str(item.get("repo", ""))),
                _repo_label(str(item.get("owner", "")), str(item.get("repo", ""))),
            )
            for item in global_targets
        ]
        + [(_REPO_ACTION_CANCEL, "Cancel")],
        cancel_message=cancel_message,
    )
    if token == _REPO_ACTION_CANCEL:
        raise SystemExit(cancel_message)
    owner, repo = _parse_repo_full_name(token)
    for item in global_targets:
        if str(item.get("owner")) == owner and str(item.get("repo")) == repo:
            return dict(item)
    raise SystemExit(cancel_message)


def _pick_existing_remote_repo(settings: dict[str, Any], *, owner_override: str | None = None, cancel_message: str) -> dict[str, Any]:
    owner = str(owner_override or _default_owner(settings)).strip()
    repos = _list_github_repos(owner)
    if not repos:
        raise SystemExit(f"No GitHub repos found for {owner}.")
    token = _pick_one(
        "Select GitHub repo",
        "Select one existing GitHub repo.",
        [(item, item) for item in repos] + [(_REPO_ACTION_CANCEL, "Cancel")],
        cancel_message=cancel_message,
    )
    if token == _REPO_ACTION_CANCEL:
        raise SystemExit(cancel_message)
    selected_owner, selected_repo = _parse_repo_full_name(token)
    metadata = github_repo_metadata(selected_owner, selected_repo) or {
        "owner": selected_owner,
        "repo": selected_repo,
        "branch": _default_branch(settings),
        "visibility": _default_visibility(settings),
    }
    return {
        "kind": "github",
        "owner": selected_owner,
        "repo": selected_repo,
        "branch": str(metadata.get("branch", "")).strip() or _default_branch(settings),
        "visibility": str(metadata.get("visibility", "")).strip().lower() or _default_visibility(settings),
    }


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


def _print_capsule_context(capsule_root: Path) -> None:
    print_block(
        "Capsule",
        (
            ("name", _capsule_id_for_path(capsule_root)),
            ("path", str(capsule_root.expanduser().resolve())),
        ),
    )


def _print_target_context(group: dict[str, Any]) -> None:
    print_block(
        "GitHub target",
        (
            ("target", _repo_label(str(group.get("owner", "")), str(group.get("repo", "")))),
            ("capsules", ", ".join(list(group.get("capsules", []))) or "-"),
        ),
    )


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
            kind = str(row.get("kind", "")).strip().lower()
            owner = str(row.get("owner", "")).strip()
            repo = str(row.get("repo", "")).strip()
            if kind == "github" and owner and repo:
                global_target = get_global_target(owner, repo, kind=kind)
                if global_target is not None:
                    row["branch"] = str(global_target.get("branch", "")).strip() or str(row.get("branch", "")).strip() or "main"
                    row["visibility"] = (
                        str(global_target.get("visibility", "")).strip().lower()
                        or str(row.get("visibility", "")).strip().lower()
                        or "private"
                    )
            row["name"] = str(name)
            row["capsule_path"] = str(root)
            row["capsule_id"] = capsule_id
            row["default"] = str(name) in set(str(item).strip() for item in list(entry.get("default_targets", []) or []))
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


def _group_repo_rows(rows: Sequence[dict[str, Any]], *, include_unattached: bool = True) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    if include_unattached:
        for target in list_global_targets():
            owner = str(target.get("owner", "")).strip()
            repo = str(target.get("repo", "")).strip()
            if not owner or not repo:
                continue
            grouped[(owner, repo)] = {
                "owner": owner,
                "repo": repo,
                "kind": str(target.get("kind", "")).strip() or "github",
                "branch": str(target.get("branch", "")).strip() or "main",
                "visibility": str(target.get("visibility", "")).strip() or "private",
                "capsules": [],
                "attachments": [],
            }
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
                "kind": str(row.get("kind", "")).strip() or "github",
                "branch": str(row.get("branch", "")).strip() or "main",
                "visibility": str(row.get("visibility", "")).strip() or "private",
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
        raise SystemExit("No configured GitHub targets were found.")
    token = _pick_one(
        title,
        "Select one GitHub target.",
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
        raise SystemExit("No target attachment was found.")
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


def _resolve_attachment_for_target(*, repo_ref: str, capsule_root: Path) -> dict[str, Any]:
    owner, repo = _parse_repo_full_name(repo_ref)
    rows = _configured_target_rows(capsule_root=capsule_root)
    matches = [
        row
        for row in rows
        if str(row.get("owner", "")).strip() == owner and str(row.get("repo", "")).strip() == repo
    ]
    if not matches:
        raise SystemExit(
            f"Capsule '{_capsule_id_for_path(capsule_root)}' is not attached to {owner}/{repo}."
        )
    return dict(matches[0])


def _resolve_repo_group(repo_ref: str | None, capsule_ref: str | None, *, caps_dir: Path | None, purpose: str) -> dict[str, Any]:
    rows = _configured_target_rows(
        capsule_root=_resolve_capsule_root(capsule_ref, caps_dir=caps_dir, purpose=purpose) if capsule_ref else None
    )
    groups = _group_repo_rows(rows, include_unattached=capsule_ref is None)
    token = str(repo_ref or "").strip()
    if token:
        owner, repo = _parse_repo_full_name(token)
        matches = [group for group in groups if str(group.get("owner")) == owner and str(group.get("repo")) == repo]
        if not matches and capsule_ref:
            matches = [
                group
                for group in _group_repo_rows(_configured_target_rows(capsule_root=None), include_unattached=True)
                if str(group.get("owner")) == owner and str(group.get("repo")) == repo
            ]
        if len(matches) == 1:
            return matches[0]
        raise SystemExit(f"Unknown configured GitHub target '{owner}/{repo}'.")
    if _is_interactive_tty():
        return _pick_repo_group(groups, title="Select GitHub target", cancel_message=f"{purpose} canceled by user.")
    raise SystemExit("No GitHub target was specified. Run interactively to pick one, or pass owner/repo.")


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
    target_meta: dict[str, Any],
    folder: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    folder_token = str(folder).strip() or inspect_capsule_directory(capsule_root).get("capsule_id") or capsule_root.name
    if folder_token == "":
        raise SystemExit("Folder cannot be empty.")
    owner = str(target_meta.get("owner", "")).strip()
    repo = str(target_meta.get("repo", "")).strip()
    if not owner or not repo:
        raise SystemExit("GitHub target selection is incomplete.")
    branch = str(target_meta.get("branch", "")).strip() or _default_branch(settings)
    visibility_token = str(target_meta.get("visibility", "")).strip().lower() or _default_visibility(settings)
    if visibility_token not in {"public", "private"}:
        raise SystemExit("Visibility must be 'public' or 'private'.")
    return {
        "name": _suggest_target_name(existing_targets, owner=owner, repo=repo),
        "kind": "github",
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "path": folder_token,
        "visibility": visibility_token,
    }


def _target_action_message(action: str) -> str:
    if action == "created":
        return "GitHub target created."
    if action == "registered":
        return "GitHub target registered."
    return "GitHub target already configured."


def _register_github_target(
    *,
    owner: str,
    repo: str,
    settings: dict[str, Any],
    persist: bool,
    create_remote_repo: bool,
    visibility: str | None = None,
) -> tuple[dict[str, Any], str]:
    existing = get_global_target(owner, repo, kind="github")
    if existing is not None:
        return dict(existing), "unchanged"

    metadata = github_repo_metadata(owner, repo)
    if metadata is not None:
        payload = {
            "kind": "github",
            "owner": owner,
            "repo": repo,
            "branch": str(metadata.get("branch", "")).strip() or _default_branch(settings),
            "visibility": str(metadata.get("visibility", "")).strip().lower() or _default_visibility(settings, visibility),
        }
        target = save_global_target(payload) if persist else dict(payload)
        return target, "registered"

    visibility_token = str(visibility or _default_visibility(settings)).strip().lower() or "private"
    if visibility_token not in {"public", "private"}:
        raise SystemExit("Visibility must be 'public' or 'private'.")
    payload = {
        "kind": "github",
        "owner": owner,
        "repo": repo,
        "branch": _default_branch(settings),
        "visibility": visibility_token,
    }
    if not create_remote_repo:
        return (save_global_target(payload) if persist else dict(payload)), "registered"
    create_repo(owner, repo, visibility_token)
    target = save_global_target(payload) if persist else dict(payload)
    return target, "created"


def create_shared_repo_interactively(
    *,
    settings: dict[str, Any],
    owner_override: str | None = None,
    repo_override: str | None = None,
    visibility_override: str | None = None,
    persist: bool = True,
    create_remote_repo: bool = True,
    show_summary: bool = True,
    skip_prompts: bool = False,
) -> dict[str, Any]:
    owner_repo_default = _repo_label(
        str(owner_override or _default_owner(settings)).strip(),
        str(repo_override or "lelabo-capsules").strip() or "lelabo-capsules",
    )
    if skip_prompts:
        owner, repo = _parse_repo_full_name(owner_repo_default)
    else:
        owner, repo = _prompt_owner_repo(settings, owner_repo_override=owner_repo_default)
    existing = get_global_target(owner, repo, kind="github")
    if existing is not None:
        target = dict(existing)
        target["action"] = "unchanged"
        target["created_repo"] = False
        return target

    remote = github_repo_metadata(owner, repo)
    if remote is not None:
        target = save_global_target(remote) if persist else dict(remote)
        target["action"] = "registered"
        target["created_repo"] = False
        return target

    visibility = _default_visibility(settings, visibility_override) if skip_prompts else _prompt_with_default("Visibility", _default_visibility(settings, visibility_override))
    visibility = str(visibility).strip().lower() or "private"
    if visibility not in {"public", "private"}:
        raise SystemExit("Visibility must be 'public' or 'private'.")
    if show_summary:
        print_block(
            "GitHub target",
            (
                ("target", _repo_label(owner, repo)),
                ("visibility", visibility),
            ),
        )
    question = "Create target?" if create_remote_repo else "Register this target?"
    if not skip_prompts and not _confirm(question, default=True):
        raise SystemExit("Target creation canceled by user.")
    target, action = _register_github_target(
        owner=owner,
        repo=repo,
        settings=settings,
        persist=persist,
        create_remote_repo=create_remote_repo,
        visibility=visibility,
    )
    target["action"] = action
    target["created_repo"] = action == "created"
    return target


def _attach_capsule_to_target(
    capsule_root: Path,
    *,
    selected_target: dict[str, Any],
    settings: dict[str, Any],
    persist: bool,
    make_default_default: bool,
    show_summary: bool,
    folder_override: str | None = None,
    default_override: bool | None = None,
    confirm_attach: bool = True,
) -> dict[str, Any]:
    info = inspect_capsule_directory(capsule_root)
    capsule_id = str(info.get("capsule_id", capsule_root.name)).strip() or capsule_root.name
    existing_targets = [item for item in list_configured_targets(capsule_root) if str(item.get("kind", "")).strip() == "github"]
    folder = str(folder_override).strip() if folder_override is not None else _prompt_with_default("Folder", capsule_id)
    set_default = bool(default_override) if default_override is not None else _confirm("Use this as a default target for this capsule?", default=make_default_default)
    target = _build_target_payload(
        capsule_root=capsule_root,
        existing_targets=existing_targets,
        target_meta=selected_target,
        folder=folder,
        settings=settings,
    )
    if show_summary:
        print_block(
            "Publish target",
            (
                ("capsule", capsule_id),
                ("target", _repo_label(str(target.get("owner", "")), str(target.get("repo", "")))),
                ("folder", target.get("path")),
                ("visibility", str(selected_target.get("visibility", "")).strip().lower() or str(target.get("visibility", ""))),
                ("default", set_default),
            ),
        )
    question = "Attach this capsule to this target?" if persist else "Use this target in preview?"
    if confirm_attach and not _confirm(question, default=True):
        raise SystemExit("Target setup canceled by user.")
    if persist:
        return save_configured_target(capsule_root, target, make_default=set_default)
    ephemeral = dict(target)
    ephemeral["default"] = False
    ephemeral["last_used"] = False
    ephemeral["_ephemeral"] = True
    return ephemeral


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
    folder_override: str | None = None,
    default_override: bool | None = None,
    skip_prompts: bool = False,
    confirm_attach: bool = True,
) -> dict[str, Any]:
    if show_summary:
        _print_capsule_context(capsule_root)

    selected_target: dict[str, Any] | None = None
    selected_owner = str(owner_override or "").strip()
    selected_repo = str(repo_override or "").strip()
    if selected_owner and selected_repo:
        existing = get_global_target(selected_owner, selected_repo, kind="github")
        if existing is not None:
            selected_target = existing
        else:
            metadata = github_repo_metadata(selected_owner, selected_repo)
            if metadata is not None:
                selected_target = save_global_target(metadata) if persist else dict(metadata)
            else:
                selected_target = create_shared_repo_interactively(
                    settings=settings,
                    owner_override=selected_owner,
                    repo_override=selected_repo,
                    visibility_override=visibility_override,
                    persist=persist,
                    create_remote_repo=create_remote_repo is not False,
                    show_summary=show_summary,
                    skip_prompts=skip_prompts,
                )
    else:
        action = _pick_one(
            "Select target source",
            "Choose how to attach this capsule.",
            (
                (_REPO_ACTION_EXISTING, "Existing configured target"),
                (_REPO_ACTION_EXISTING_REMOTE, "Existing GitHub repo"),
                (_REPO_ACTION_CREATE, "Create new GitHub target"),
                (_REPO_ACTION_CANCEL, "Cancel"),
            ),
            cancel_message="Target setup canceled by user.",
        )
        if action == _REPO_ACTION_CANCEL:
            raise SystemExit("Target setup canceled by user.")
        if action == _REPO_ACTION_EXISTING:
            selected_target = _pick_existing_target(list_global_targets(), cancel_message="Target selection canceled by user.")
        elif action == _REPO_ACTION_EXISTING_REMOTE:
            selected_target = _pick_existing_remote_repo(settings, owner_override=owner_override, cancel_message="Target selection canceled by user.")
            existing = get_global_target(str(selected_target.get("owner", "")), str(selected_target.get("repo", "")), kind="github")
            if existing is not None:
                selected_target = existing
            else:
                selected_target = save_global_target(selected_target) if persist else dict(selected_target)
        else:
            selected_target = create_shared_repo_interactively(
                settings=settings,
                owner_override=owner_override,
                repo_override=repo_override,
                visibility_override=visibility_override,
                persist=persist,
                create_remote_repo=create_remote_repo is not False,
                show_summary=show_summary,
                skip_prompts=skip_prompts,
            )

    if selected_target is None:
        raise SystemExit("GitHub target selection is incomplete.")

    return _attach_capsule_to_target(
        capsule_root,
        selected_target=selected_target,
        settings=settings,
        persist=persist,
        make_default_default=make_default_default,
        show_summary=show_summary,
        folder_override=folder_override,
        default_override=default_override,
        confirm_attach=confirm_attach,
    )


def _pick_repo_capsule(rows: Sequence[dict[str, Any]], *, title: str, cancel_message: str) -> dict[str, Any]:
    if not rows:
        raise SystemExit("No capsule was found in this target repo.")
    if len(rows) == 1:
        return dict(rows[0])
    token = _pick_one(
        title,
        "Select one capsule from the target repo.",
        [
            (
                f"{row.get('capsule_id')}::{row.get('folder')}",
                f"{row.get('capsule_id')} | folder: {row.get('folder')}",
            )
            for row in rows
        ],
        cancel_message=cancel_message,
    )
    capsule_id, _, folder = token.partition("::")
    for row in rows:
        if str(row.get("capsule_id")) == capsule_id and str(row.get("folder")) == folder:
            return dict(row)
    raise SystemExit(cancel_message)


def _cmd_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo targets list",
        description="List configured GitHub targets.",
    )
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule id or alias")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="Target listing") if args.capsule_ref else None
    groups = _group_repo_rows(_configured_target_rows(capsule_root=capsule_root), include_unattached=capsule_root is None)
    if bool(args.json):
        _print_json(
            {
                "schema_version": TARGETS_JSON_SCHEMA,
                "command": "list",
                "capsule_path": str(capsule_root) if capsule_root is not None else None,
                "targets": groups,
            }
        )
    else:
        if not groups:
            print_status("info", "No configured GitHub targets were found.")
            return 0
        print("GitHub targets")
        for group in groups:
            print(f"- {group.get('owner', '-')}/{group.get('repo', '-')}")
            capsules = ", ".join(group.get("capsules", [])) or "-"
            print(f"  capsules: {capsules}")
    return 0


def _cmd_create(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo targets create",
        description="Create or register one shared GitHub target.",
    )
    parser.add_argument("--target", default=None, help="GitHub target as owner/repo")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="Create or register a public GitHub target")
    vis.add_argument("--private", action="store_true", help="Create or register a private GitHub target")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    settings = _effective_settings()
    visibility = "public" if bool(args.public) else "private" if bool(args.private) else None
    owner_override, repo_override = (None, None)
    if args.target:
        owner_override, repo_override = _parse_repo_full_name(args.target)
    repo_info = create_shared_repo_interactively(
        settings=settings,
        owner_override=owner_override,
        repo_override=repo_override,
        visibility_override=visibility,
        persist=True,
        create_remote_repo=True,
        show_summary=not bool(args.json),
    )
    if bool(args.json):
        _print_json(
            {
                "schema_version": TARGET_JSON_SCHEMA,
                "command": "create",
                "action": repo_info.get("action"),
                "target": repo_info,
            }
        )
    else:
        action = str(repo_info.get("action", "")).strip() or "created"
        print_status("success", _target_action_message(action))
        print_block("Target", (("target", _repo_label(repo_info["owner"], repo_info["repo"])), ("visibility", repo_info["visibility"])))
    return 0


def _cmd_attach(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo targets attach",
        description="Attach one capsule to a configured GitHub target.",
    )
    parser.add_argument("repo_ref", help="Configured GitHub target as owner/repo")
    parser.add_argument("capsule_ref", help="Visible capsule id, alias, or local capsule path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    group = _resolve_repo_group(args.repo_ref, None, caps_dir=caps_dir, purpose="Target attach")
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="Target attach")
    if not bool(args.json):
        _print_target_context(group)
        _print_capsule_context(capsule_root)
    target_repo = {
        "kind": str(group.get("kind", "")).strip() or "github",
        "owner": str(group.get("owner", "")).strip(),
        "repo": str(group.get("repo", "")).strip(),
        "branch": str(group.get("branch", "")).strip() or _default_branch(settings),
        "visibility": str(group.get("visibility", "")).strip().lower() or _default_visibility(settings),
    }
    target = _attach_capsule_to_target(
        capsule_root,
        selected_target=target_repo,
        settings=settings,
        persist=True,
        make_default_default=True,
        show_summary=not bool(args.json),
    )
    payload = {
        "schema_version": TARGET_JSON_SCHEMA,
        "command": "attach",
        "capsule_path": str(capsule_root),
        "target": target,
    }
    if bool(args.json):
        _print_json(payload)
    else:
        print_status("success", "Capsule attached to target.")
        print_block(
            "Target",
            (
                ("capsule", _capsule_id_for_path(capsule_root)),
                ("target", _repo_label(str(target.get("owner", "")), str(target.get("repo", "")))),
                ("folder", target.get("path")),
                ("default", target.get("default")),
            ),
        )
    return 0


def _cmd_defaults(argv: list[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "Manage default publish targets for one capsule.\n\n"
            "Usage:\n"
            "  lelabo targets defaults list <capsule>\n"
            "  lelabo targets defaults add <capsule> <target>\n"
            "  lelabo targets defaults remove <capsule> <target>\n"
        )
        return 0
    cmd = str(args[0]).strip().lower()
    rest = args[1:]
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)

    if cmd == "list":
        parser = argparse.ArgumentParser(prog="lelabo targets defaults list")
        parser.add_argument("capsule_ref", help="Visible capsule id, alias, or local capsule path")
        parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
        parsed = parser.parse_args(rest)
        capsule_root = _resolve_capsule_root(parsed.capsule_ref, caps_dir=caps_dir, purpose="Target defaults")
        rows = [row for row in _configured_target_rows(capsule_root=capsule_root) if bool(row.get("default"))]
        payload = {
            "schema_version": TARGETS_JSON_SCHEMA,
            "command": "defaults-list",
            "capsule_path": str(capsule_root),
            "targets": [
                {
                    "target": _repo_label(str(row.get("owner", "")), str(row.get("repo", ""))),
                    "name": row.get("name"),
                    "folder": row.get("path"),
                }
                for row in rows
            ],
        }
        if bool(parsed.json):
            _print_json(payload)
        else:
            if not rows:
                print_status("info", "This capsule has no default targets.")
            else:
                print_block("Capsule", (("name", _capsule_id_for_path(capsule_root)),))
                for row in rows:
                    print(f"- {_repo_label(str(row.get('owner', '')), str(row.get('repo', '')))} | folder: {row.get('path')}")
        return 0

    parser = argparse.ArgumentParser(prog=f"lelabo targets defaults {cmd}")
    parser.add_argument("capsule_ref", help="Visible capsule id, alias, or local capsule path")
    parser.add_argument("repo_ref", help="Configured GitHub target as owner/repo")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parsed = parser.parse_args(rest)
    capsule_root = _resolve_capsule_root(parsed.capsule_ref, caps_dir=caps_dir, purpose="Target defaults")
    attachment = _resolve_attachment_for_target(repo_ref=parsed.repo_ref, capsule_root=capsule_root)
    if cmd == "add":
        target = set_default_target(capsule_root, str(attachment.get("name", "")))
        success_message = "Default targets updated."
        command_name = "defaults-add"
    elif cmd == "remove":
        target = unset_default_target(capsule_root, str(attachment.get("name", "")))
        success_message = "Default targets updated."
        command_name = "defaults-remove"
    else:
        raise SystemExit("Unknown targets defaults subcommand. Use one of: list, add, remove.")
    payload = {
        "schema_version": TARGET_JSON_SCHEMA,
        "command": command_name,
        "capsule_path": str(capsule_root),
        "target": target,
    }
    if bool(parsed.json):
        _print_json(payload)
    else:
        print_status("success", success_message)
        print_block(
            "Target",
            (
                ("capsule", _capsule_id_for_path(capsule_root)),
                ("target", _repo_label(str(attachment.get("owner", "")), str(attachment.get("repo", "")))),
                ("folder", attachment.get("path")),
                ("default", target.get("default")),
            ),
        )
    return 0


def _cmd_import(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo targets import",
        description="Import capsule(s) from a configured target repo.",
    )
    parser.add_argument("repo_ref", help="Configured GitHub target as owner/repo")
    parser.add_argument("--capsule", default=None, help="Specific capsule id to import from the target repo")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    group = _resolve_repo_group(args.repo_ref, None, caps_dir=caps_dir, purpose="Target import")
    target_repo = {
        "kind": str(group.get("kind", "")).strip() or "github",
        "owner": str(group.get("owner", "")).strip(),
        "repo": str(group.get("repo", "")).strip(),
        "branch": str(group.get("branch", "")).strip() or _default_branch(settings),
        "visibility": str(group.get("visibility", "")).strip().lower() or _default_visibility(settings),
    }
    if not bool(args.json):
        _print_target_context(group)
    repo_capsules = inspect_target_repo_capsules(target_repo)
    if args.capsule:
        matches = [row for row in repo_capsules if str(row.get("capsule_id", "")).strip() == str(args.capsule).strip()]
        if not matches:
            raise SystemExit(f"Capsule '{args.capsule}' was not found in {args.repo_ref}.")
        imported = dict(matches[0])
    elif len(repo_capsules) == 1:
        imported = dict(repo_capsules[0])
    elif _is_interactive_tty():
        imported = _pick_repo_capsule(repo_capsules, title="Select target repo capsule", cancel_message="Target import canceled by user.")
    else:
        raise SystemExit("This target repo contains multiple capsules. Re-run interactively or pass `--capsule <capsule_id>`.")
    try:
        entry = install_capsule_from_directory(
            source_dir=Path(str(imported.get("root", ""))).expanduser().resolve(),
            capsules_dir=caps_dir,
            source_meta={
                "type": "github_repo",
                "path": _repo_label(str(group.get("owner", "")), str(group.get("repo", ""))),
                "target": _repo_label(str(group.get("owner", "")), str(group.get("repo", ""))),
                "folder": str(imported.get("folder", "")),
            },
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to import capsule from {_repo_label(str(group.get('owner', '')), str(group.get('repo', '')))}.\n{exc}"
        ) from exc
    payload = {
        "schema_version": TARGET_JSON_SCHEMA,
        "command": "import",
        "target": {
            "owner": group.get("owner"),
            "repo": group.get("repo"),
            "imported_capsule_id": entry.get("capsule_id"),
            "install_action": entry.get("install_action"),
        },
    }
    if bool(args.json):
        _print_json(payload)
    else:
        print_status("success", "Capsule imported from target repo.")
        print_block(
            "Target",
            (
                ("capsule", entry.get("capsule_id")),
                ("target", _repo_label(str(group.get("owner", "")), str(group.get("repo", "")))),
                ("folder", imported.get("folder")),
            ),
        )
    return 0


def _cmd_remove_capsule(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo targets remove-capsule",
        description="Remove one capsule from a target repo and detach it locally.",
    )
    parser.add_argument("repo_ref", help="Configured GitHub target as owner/repo")
    parser.add_argument("capsule_ref", help="Visible capsule id, alias, or local capsule path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="Target remove-capsule")
    attachment = _resolve_attachment_for_target(repo_ref=args.repo_ref, capsule_root=capsule_root)
    if not _confirm(
        f"Remove {_capsule_id_for_path(capsule_root)} from {args.repo_ref}?",
        default=False,
    ):
        raise SystemExit("Target remove-capsule canceled by user.")
    removed = remove_target_repo_capsule(
        target={
            "kind": str(attachment.get("kind", "")).strip() or "github",
            "owner": str(attachment.get("owner", "")).strip(),
            "repo": str(attachment.get("repo", "")).strip(),
            "branch": str(attachment.get("branch", "")).strip() or _default_branch(settings),
            "visibility": str(attachment.get("visibility", "")).strip().lower() or _default_visibility(settings),
        },
        folder=str(attachment.get("path", "")),
    )
    target = remove_configured_target(capsule_root, str(attachment.get("name", "")))
    if isinstance(target, dict):
        target["remote_remove"] = removed
    payload = {
        "schema_version": TARGET_JSON_SCHEMA,
        "command": "remove-capsule",
        "capsule_path": str(capsule_root),
        "target": target,
    }
    if bool(args.json):
        _print_json(payload)
    else:
        print_status("success", "Capsule removed from target repo.")
        print_block(
            "Target",
            (
                ("capsule", _capsule_id_for_path(capsule_root)),
                ("target", args.repo_ref),
                ("folder", attachment.get("path")),
                ("default", target.get("default") if isinstance(target, dict) else None),
            ),
        )
    return 0


def _cmd_delete(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo targets delete",
        description="Delete one local target entry and clean its attachments.",
    )
    parser.add_argument("repo_ref", help="Configured GitHub target as owner/repo")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    group = _resolve_repo_group(args.repo_ref, None, caps_dir=caps_dir, purpose="Target edit")
    if not bool(args.json):
        _print_target_context(group)
    target_name = _repo_label(str(group.get("owner", "")), str(group.get("repo", "")))
    capsules = ", ".join(list(group.get("capsules", []))) or "-"
    if not _confirm(f"Delete local target {target_name}? Attached capsules: {capsules}.", default=False):
        raise SystemExit("Target delete canceled by user.")
    target = remove_global_target(
        owner=str(group.get("owner", "")).strip(),
        repo=str(group.get("repo", "")).strip(),
        kind=str(group.get("kind", "")).strip() or "github",
    )
    payload = {
        "schema_version": TARGET_JSON_SCHEMA,
        "command": "delete",
        "target": target,
    }
    if bool(args.json):
        _print_json(payload)
    else:
        print_status("success", "Target deleted from local catalog.")
        print_block("Target", (("target", target_name), ("capsules", capsules)))
    return 0


def _cmd_edit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo targets edit",
        description="Manage one target, its attached capsules, and its default-target state.",
    )
    parser.add_argument("repo_ref", nargs="?", default=None, help="Optional configured GitHub target as owner/repo")
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule id or alias")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    group = _resolve_repo_group(args.repo_ref, args.capsule_ref, caps_dir=caps_dir, purpose="Target edit")
    if not bool(args.json):
        _print_target_context(group)
    action = _pick_one(
        "Edit target",
        "Choose one action for this target.",
        (
            ("attach", "Attach capsule to this target"),
            ("default-add", "Add to default targets for a capsule"),
            ("default-remove", "Remove from default targets for a capsule"),
            ("detach", "Detach capsule"),
            ("remove-remote", "Remove capsule from target repo"),
            ("import", "Import capsule from target repo"),
            ("delete-target", "Delete target"),
            (_REPO_ACTION_CANCEL, "Cancel"),
        ),
        cancel_message="Target edit canceled by user.",
    )
    if action == _REPO_ACTION_CANCEL:
        raise SystemExit("Target edit canceled by user.")
    selected: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    command_name = "edit"
    success_message = "Target updated."
    target_repo = {
        "kind": str(group.get("kind", "")).strip() or "github",
        "owner": str(group.get("owner", "")).strip(),
        "repo": str(group.get("repo", "")).strip(),
        "branch": str(group.get("branch", "")).strip() or _default_branch(settings),
        "visibility": str(group.get("visibility", "")).strip().lower() or _default_visibility(settings),
    }

    if action == "attach":
        if not args.capsule_ref:
            raise SystemExit("Missing capsule. Use `lelabo targets edit <owner/repo> <capsule>`, then choose 'Attach capsule to this target'.")
        capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="edit")
        if not bool(args.json):
            _print_target_context(group)
            _print_capsule_context(capsule_root)
        target = _attach_capsule_to_target(
            capsule_root,
            selected_target=target_repo,
            settings=settings,
            persist=True,
            make_default_default=True,
            show_summary=not bool(args.json),
        )
        selected = {
            "capsule_id": _capsule_id_for_path(capsule_root),
            "owner": target.get("owner"),
            "repo": target.get("repo"),
            "path": target.get("path"),
        }
        command_name = "edit-attach"
        success_message = "Capsule attached to target."
    elif action == "import":
        if not bool(args.json):
            _print_target_context(group)
        repo_capsules = inspect_target_repo_capsules(target_repo)
        imported = _pick_repo_capsule(repo_capsules, title="Select target repo capsule", cancel_message="Target edit canceled by user.")
        try:
            entry = install_capsule_from_directory(
                source_dir=Path(str(imported.get("root", ""))).expanduser().resolve(),
                capsules_dir=caps_dir,
                source_meta={
                    "type": "github_repo",
                    "path": _repo_label(str(group.get("owner", "")), str(group.get("repo", ""))),
                    "target": _repo_label(str(group.get("owner", "")), str(group.get("repo", ""))),
                    "folder": str(imported.get("folder", "")),
                },
            )
        except Exception as exc:
            raise SystemExit(
                f"Failed to import capsule from {_repo_label(str(group.get('owner', '')), str(group.get('repo', '')))}.\n{exc}"
            ) from exc
        target = {
            "owner": group.get("owner"),
            "repo": group.get("repo"),
            "imported_capsule_id": entry.get("capsule_id"),
            "install_action": entry.get("install_action"),
        }
        selected = {
            "capsule_id": entry.get("capsule_id"),
            "owner": group.get("owner"),
            "repo": group.get("repo"),
            "path": imported.get("folder"),
        }
        command_name = "edit-import"
        success_message = "Capsule imported from target repo."
    elif action == "delete-target":
        if not bool(args.json):
            _print_target_context(group)
        target_name = _repo_label(str(group.get("owner", "")), str(group.get("repo", "")))
        capsules = ", ".join(list(group.get("capsules", []))) or "-"
        if not _confirm(f"Delete local target {target_name}? Attached capsules: {capsules}.", default=False):
            raise SystemExit("Target edit canceled by user.")
        try:
            target = remove_global_target(
                owner=str(group.get("owner", "")).strip(),
                repo=str(group.get("repo", "")).strip(),
                kind=str(group.get("kind", "")).strip() or "github",
            )
        except ValueError as exc:
            raise SystemExit(str(exc))
        selected = {
            "capsule_id": capsules,
            "owner": group.get("owner"),
            "repo": group.get("repo"),
            "path": "-",
        }
        command_name = "edit-delete-target"
        success_message = "Target deleted from local catalog."
    else:
        attachments = list(group.get("attachments", []))
        if not attachments:
            raise SystemExit("No capsule is attached to this target.")
        if not bool(args.json):
            _print_target_context(group)
        if args.capsule_ref:
            filter_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="edit")
            attachments = [
                row for row in attachments if Path(str(row.get("capsule_path", ""))).expanduser().resolve() == filter_root
            ]
            if not attachments:
                raise SystemExit(
                    f"Capsule '{_capsule_id_for_path(filter_root)}' is not attached to {_repo_label(str(group.get('owner', '')), str(group.get('repo', '')))}."
                )
        selected = _pick_attachment(
            attachments,
            title="Select capsule",
            cancel_message="Target edit canceled by user.",
        )
        capsule_root = Path(str(selected.get("capsule_path", ""))).expanduser().resolve()
        if action == "default-add":
            try:
                target = set_default_target(capsule_root, str(selected.get("name", "")))
            except ValueError as exc:
                raise SystemExit(str(exc))
            command_name = "edit-default-add"
            success_message = "Default targets updated."
        elif action == "default-remove":
            try:
                target = unset_default_target(capsule_root, str(selected.get("name", "")))
            except ValueError as exc:
                raise SystemExit(str(exc))
            command_name = "edit-default-remove"
            success_message = "Default targets updated."
        elif action == "detach":
            try:
                target = remove_configured_target(capsule_root, str(selected.get("name", "")))
            except ValueError as exc:
                raise SystemExit(str(exc))
            command_name = "edit-detach"
            success_message = "Capsule detached from target."
        elif action == "remove-remote":
            if not _confirm(
                f"Remove {selected.get('capsule_id')} from {_repo_label(str(group.get('owner', '')), str(group.get('repo', '')))}?",
                default=False,
            ):
                raise SystemExit("Target edit canceled by user.")
            try:
                removed = remove_target_repo_capsule(target=target_repo, folder=str(selected.get("path", "")))
            except ValueError as exc:
                raise SystemExit(str(exc))
            detach_locally = _confirm("Also detach this capsule locally?", default=True)
            if detach_locally:
                try:
                    target = remove_configured_target(capsule_root, str(selected.get("name", "")))
                except ValueError as exc:
                    raise SystemExit(str(exc))
            else:
                target = {
                    "name": selected.get("name"),
                    "owner": selected.get("owner"),
                    "repo": selected.get("repo"),
                    "path": selected.get("path"),
                    "default": bool(selected.get("default")),
                    "detached": False,
                }
            if isinstance(target, dict):
                target["remote_remove"] = removed
            command_name = "edit-remove-remote"
            success_message = "Capsule removed from target repo."
        else:
            raise SystemExit("Target edit canceled by user.")
    if bool(args.json):
        _print_json(
            {
                "schema_version": TARGET_JSON_SCHEMA,
                "command": command_name,
                "capsule_path": str(capsule_root) if 'capsule_root' in locals() else None,
                "target": target,
            }
        )
    else:
        print_status("success", success_message)
        print_block(
            "Target",
            (
                ("capsule", selected.get("capsule_id")),
                ("target", _repo_label(str(selected.get("owner", "")), str(selected.get("repo", "")))),
                ("folder", selected.get("path")),
                ("default", target.get("default") if isinstance(target, dict) else None),
            ),
        )
    return 0


def _cmd_detach(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo targets detach",
        description="Detach one capsule from a configured GitHub target.",
    )
    parser.add_argument("repo_ref", help="Configured GitHub target as owner/repo")
    parser.add_argument("capsule_ref", help="Visible capsule id, alias, or local capsule path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    group = _resolve_repo_group(args.repo_ref, args.capsule_ref, caps_dir=caps_dir, purpose="Target detach")
    attachments = list(group.get("attachments", []))
    if args.capsule_ref:
        capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir, purpose="Target detach")
        attachments = [row for row in attachments if Path(str(row.get("capsule_path", ""))).expanduser().resolve() == capsule_root]
        if not attachments:
            raise SystemExit(
                f"Capsule '{_capsule_id_for_path(capsule_root)}' is not attached to {_repo_label(str(group.get('owner', '')), str(group.get('repo', '')))}."
            )
    selected = _pick_attachment(attachments, title="Select capsule", cancel_message="Target detach canceled by user.")
    repo_name = _repo_label(str(selected.get("owner", "")), str(selected.get("repo", "")))
    if not _confirm(f"Detach this capsule from {repo_name}?", default=False):
        raise SystemExit("Target detach canceled by user.")
    capsule_root = Path(str(selected.get("capsule_path", ""))).expanduser().resolve()
    try:
        target = remove_configured_target(capsule_root, str(selected.get("name", "")))
    except ValueError as exc:
        raise SystemExit(str(exc))
    if bool(args.json):
        _print_json(
            {
                "schema_version": TARGET_JSON_SCHEMA,
                "command": "detach",
                "capsule_path": str(capsule_root),
                "target": target,
            }
        )
    else:
        print_status("success", "Capsule detached from target.")
        print_block(
            "Target",
            (
                ("capsule", selected.get("capsule_id")),
                ("target", repo_name),
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
    if cmd == "attach":
        return _cmd_attach(rest)
    if cmd == "defaults":
        return _cmd_defaults(rest)
    if cmd == "import":
        return _cmd_import(rest)
    if cmd == "remove-capsule":
        return _cmd_remove_capsule(rest)
    if cmd == "delete":
        return _cmd_delete(rest)
    if cmd == "edit":
        return _cmd_edit(rest)
    if cmd == "detach":
        return _cmd_detach(rest)
    raise SystemExit(
        f"Unknown targets subcommand: {cmd}\n\n"
        "Use one of: list, create, attach, detach, defaults, import, remove-capsule, delete.\n"
        "Run `lelabo targets -h` for usage."
    )


__all__ = [
    "configure_github_target_for_capsule",
    "create_shared_repo_interactively",
    "main",
]
