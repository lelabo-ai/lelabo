"""Capsule-first publish command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ...capsule import get_capsule, inspect_capsule_directory
from ...capsule.discovery import DiscoveredCapsule, discover_workspace_capsules, find_capsule_root, is_capsule_root
from ...capsule.publish import (
    auto_commit_message,
    available_targets,
    current_github_login,
    get_target_preferences,
    push_github_target,
    resolve_target,
    save_configured_target,
)
from ...config.user_settings import load_effective_settings
from ..interactive_picker import pick_many_with_checkboxes
from .repo import configure_github_target_for_capsule
from ..ui import print_block, print_list_block, print_status


PUSH_JSON_SCHEMA = "lelabo.cli.push/v1"
PUSHES_JSON_SCHEMA = "lelabo.cli.pushes/v1"
_CREATE_TARGET_OPTION = "__create_new_github_target__"


PUSH_HELP = """\
Publish one capsule to configured remote targets.

Usage:
  lelabo push [capsule_ref] [--target NAME] [--all-targets] [-m MESSAGE] [--preview] [--yes] [--json]

Notes:
  - `capsule_ref` can be a local path or a stored capsule id/alias
  - `lelabo push` publishes to saved remote targets
  - use `lelabo export` for local exports
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


def _prompt_choice(title: str, options: Sequence[tuple[str, str]], *, default: str) -> str:
    if str(title).strip():
        print(title)
    for key, label in options:
        print(f"{key}. {label}")
    valid = {str(key).strip(): str(key).strip() for key, _ in options if str(key).strip()}
    fallback = str(default).strip()
    while True:
        answer = input(f"Choice [{fallback}]: ").strip() or fallback
        if answer in valid:
            return valid[answer]
        print_status("warning", "Enter a valid choice.")


def _pick_capsule_interactively(candidates: Sequence[DiscoveredCapsule], *, title: str) -> Path:
    options = [(item.path, f"{item.capsule_id} | path: {item.path}") for item in candidates]
    selected = pick_many_with_checkboxes(
        title=title,
        text="Select one capsule before choosing publish targets.",
        options=options,
        selection_noun="capsule",
        confirm_button_text="Use selected",
        max_selection_count=1,
        max_selection_message="Select exactly one capsule before confirming.",
    )
    if selected is None:
        raise SystemExit("Push canceled by user.")
    chosen = str(selected[0]).strip() if selected else ""
    for item in candidates:
        if item.path == chosen:
            return item.root
    raise SystemExit("Push canceled by user.")


def _resolve_capsule_root(capsule_ref: str | None, *, caps_dir: Path | None) -> Path:
    if not capsule_ref:
        candidates = list(discover_workspace_capsules(start=Path.cwd()))
        if len(candidates) == 1:
            return candidates[0].root
        if not candidates:
            raise SystemExit(
                "No capsule found in the current workspace. Pass a capsule path/id or create a capsule under this directory."
            )
        if _is_interactive_tty():
            return _pick_capsule_interactively(candidates, title="Select capsule to push")
        names = ", ".join(item.capsule_id for item in candidates)
        raise SystemExit(
            f"Multiple capsules found in the current workspace: {names}. Pass a capsule path/id to `lelabo push`."
        )

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


def _repo_label(target: dict[str, Any]) -> str:
    owner = str(target.get("owner", "")).strip()
    repo = str(target.get("repo", "")).strip()
    if owner and repo:
        return f"{owner}/{repo}"
    return repo or "-"


def _target_picker_label(target: dict[str, Any]) -> str:
    kind = str(target.get("kind", "")).strip() or "-"
    folder = str(target.get("path", "")).strip() or "-"
    visibility = str(target.get("visibility", "")).strip() or "private"
    return f"{kind} | repo: {_repo_label(target)} | folder: {folder} | visibility: {visibility}"


def _target_summary(target: dict[str, Any]) -> str:
    kind = str(target.get("kind", "")).strip() or "-"
    folder = str(target.get("path", "")).strip() or "-"
    return f"{kind} | repo: {_repo_label(target)} | folder: {folder}"


def _push_targets(capsule_root: Path) -> list[dict[str, Any]]:
    return [item for item in available_targets(capsule_root) if str(item.get("kind", "")).strip() != "workspace"]


def _find_target_by_name(targets: Sequence[dict[str, Any]], name: str) -> dict[str, Any] | None:
    token = str(name).strip()
    for item in targets:
        if str(item.get("name", "")).strip() == token:
            return item
    return None


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


def _default_github_target(
    *,
    capsule_root: Path,
    owner: str | None,
    repo: str | None,
    branch: str | None,
    visibility: str,
    existing_targets: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    capsule_id = str(inspect_capsule_directory(capsule_root).get("capsule_id", capsule_root.name)).strip() or capsule_root.name
    login = current_github_login()
    resolved_owner = str(owner or "").strip() or str(login or "").strip()
    resolved_repo = str(repo or "").strip() or "lelabo-capsules"
    return {
        "name": _suggest_target_name(list(existing_targets or []), owner=resolved_owner, repo=resolved_repo),
        "kind": "github",
        "owner": resolved_owner,
        "repo": resolved_repo,
        "branch": str(branch or "").strip() or "main",
        "path": capsule_id,
        "visibility": str(visibility).strip().lower() or "private",
    }


def _validate_github_target(target: dict[str, Any]) -> dict[str, Any]:
    name = str(target.get("name", "")).strip()
    owner = str(target.get("owner", "")).strip()
    repo = str(target.get("repo", "")).strip()
    branch = str(target.get("branch", "")).strip() or "main"
    folder = str(target.get("path", "")).strip()
    visibility = str(target.get("visibility", "")).strip().lower() or "private"
    if not name:
        raise SystemExit("Target name cannot be empty.")
    if not owner:
        raise SystemExit("GitHub owner cannot be empty.")
    if not repo:
        raise SystemExit("GitHub repo cannot be empty.")
    if not folder:
        raise SystemExit("Folder cannot be empty.")
    if visibility not in {"public", "private"}:
        raise SystemExit("Visibility must be 'public' or 'private'.")
    normalized = dict(target)
    normalized.update(
        {
            "name": name,
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "path": folder,
            "visibility": visibility,
        }
    )
    return normalized


def _materialize_github_target(
    *,
    capsule_root: Path,
    target: dict[str, Any],
    persist: bool,
    make_default: bool,
) -> dict[str, Any]:
    normalized = _validate_github_target(target)
    if persist:
        return save_configured_target(capsule_root, normalized, make_default=make_default)
    out = dict(normalized)
    out["default"] = False
    out["last_used"] = False
    out["_ephemeral"] = True
    return out


def _prompt_custom_github_target(
    *,
    capsule_root: Path,
    proposed: dict[str, Any],
    persist: bool,
    make_default: bool,
    existing_targets: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target = {
        "name": "github",
        "kind": "github",
        "owner": _prompt_with_default("Owner", str(proposed.get("owner", "")).strip()),
        "repo": _prompt_with_default("Repo", str(proposed.get("repo", "")).strip() or "lelabo-capsules"),
        "branch": _prompt_with_default("Branch", str(proposed.get("branch", "")).strip() or "main"),
        "path": _prompt_with_default("Folder", str(proposed.get("path", "")).strip()),
        "visibility": _prompt_with_default("Visibility", str(proposed.get("visibility", "")).strip() or "private"),
    }
    target["name"] = _suggest_target_name(
        list(existing_targets or []),
        owner=str(target.get("owner", "")).strip(),
        repo=str(target.get("repo", "")).strip(),
    )
    target = _validate_github_target(target)
    print(f"Proposed GitHub target: {_repo_label(target)}")
    print(f"Folder: {target['path']}")
    print(f"Visibility: {target['visibility']}")
    if not _confirm("Create it now?", default=True):
        raise SystemExit("Publish target creation canceled by user.")
    return _materialize_github_target(
        capsule_root=capsule_root,
        target=target,
        persist=persist,
        make_default=make_default,
    )


def _bootstrap_github_target(
    *,
    capsule_root: Path,
    owner: str | None,
    repo: str | None,
    branch: str | None,
    visibility: str,
    assume_yes: bool,
    preview: bool,
    make_default: bool,
    existing_targets: Sequence[dict[str, Any]] | None = None,
    show_status: bool = True,
) -> dict[str, Any]:
    proposed = _default_github_target(
        capsule_root=capsule_root,
        owner=owner,
        repo=repo,
        branch=branch,
        visibility=visibility,
        existing_targets=existing_targets,
    )

    if not assume_yes:
        if not _is_interactive_tty():
            raise SystemExit(
                "No saved publish target is available for this capsule. "
                "Run interactively to add a GitHub target, or pass `--yes` to accept the proposed GitHub target."
            )
        capsule_id = str(inspect_capsule_directory(capsule_root).get("capsule_id", capsule_root.name)).strip() or capsule_root.name
        if show_status:
            print_status("info", f"No saved publish target for {capsule_id}.")
        target = configure_github_target_for_capsule(
            capsule_root,
            settings={"github": {"owner": owner or "", "default_visibility": visibility, "default_branch": branch or "main"}},
            persist=not preview,
            owner_override=owner,
            repo_override=repo,
            visibility_override=visibility,
            make_default_default=make_default,
            create_remote_repo=not preview,
            show_summary=show_status,
        )
    else:
        if not str(proposed.get("owner", "")).strip():
            raise SystemExit(
                "GitHub owner could not be resolved. Set --owner or `lelabo config set github.owner <owner>`."
            )
        target = _materialize_github_target(
            capsule_root=capsule_root,
            target=proposed,
            persist=not preview,
            make_default=make_default,
        )

    target["bootstrapped"] = True
    return target


def _override_target_from_args(
    capsule_root: Path,
    *,
    target_name: str | None,
    owner: str | None,
    repo: str | None,
    branch: str | None,
    visibility: str,
    preview: bool,
) -> dict[str, Any] | None:
    if not any(str(item or "").strip() for item in (owner, repo, branch)):
        return None
    capsule_id = str(inspect_capsule_directory(capsule_root).get("capsule_id", capsule_root.name)).strip() or capsule_root.name
    name = str(target_name or "github").strip() or "github"
    current: dict[str, Any] = {}
    try:
        current = resolve_target(capsule_root, name)
    except Exception:
        current = {}
    target = {
        "name": name,
        "kind": "github",
        "owner": str(owner or current.get("owner") or "").strip(),
        "repo": str(repo or current.get("repo") or "lelabo-capsules").strip(),
        "branch": str(branch or current.get("branch") or "main").strip() or "main",
        "path": str(current.get("path") or capsule_id).strip() or capsule_id,
        "visibility": str(visibility or current.get("visibility") or "private").strip().lower() or "private",
    }
    return _materialize_github_target(
        capsule_root=capsule_root,
        target=target,
        persist=not preview,
        make_default=True,
    )


def _pick_targets_interactively(
    *,
    capsule_root: Path,
    targets: list[dict[str, Any]],
    owner: str | None,
    repo: str | None,
    branch: str | None,
    visibility: str,
    preview: bool,
) -> list[dict[str, Any]]:
    selected_names: set[str] = set()
    extra_targets: dict[str, dict[str, Any]] = {}

    while True:
        display_targets = list(targets)
        for name, target in extra_targets.items():
            found = _find_target_by_name(display_targets, name)
            if found is None:
                display_targets.append(target)
            else:
                display_targets[display_targets.index(found)] = target

        options = [(str(item.get("name", "")).strip(), _target_picker_label(item)) for item in display_targets]
        options.append((_CREATE_TARGET_OPTION, "Create new GitHub target"))
        selected = pick_many_with_checkboxes(
            title="Select publish targets",
            text="Choose one or more remote publish targets.",
            options=options,
            default_values=sorted(name for name in selected_names if name),
            empty_selection_message="Select at least one publish target before confirming.",
            selection_noun="target",
            confirm_button_text="Confirm selected targets",
        )
        if selected is None:
            raise SystemExit("Push canceled by user.")
        selected_names = {str(item).strip() for item in selected if str(item).strip()}
        if _CREATE_TARGET_OPTION in selected_names:
            selected_names.discard(_CREATE_TARGET_OPTION)
            proposed = _default_github_target(
                capsule_root=capsule_root,
                owner=owner,
                repo=repo,
                branch=branch,
                visibility=visibility,
                existing_targets=[*display_targets, *extra_targets.values()],
            )
            new_target = _prompt_custom_github_target(
                capsule_root=capsule_root,
                proposed=proposed,
                persist=not preview,
                make_default=False,
                existing_targets=[*display_targets, *extra_targets.values()],
            )
            if preview:
                extra_targets[str(new_target.get("name", "")).strip()] = new_target
            else:
                targets = available_targets(capsule_root)
            selected_names.add(str(new_target.get("name", "")).strip())
            continue
        chosen = [item for item in display_targets if str(item.get("name", "")).strip() in selected_names]
        if chosen:
            return chosen


def _confirm_multiple_targets(chosen: Sequence[dict[str, Any]], *, preview: bool) -> None:
    if len(chosen) <= 1:
        return
    print_list_block("Publish targets", [_target_summary(item) for item in chosen])
    question = "Continue with this preview?" if preview else "Continue?"
    if not _confirm(question, default=True):
        raise SystemExit("Push canceled by user.")


def push_capsule(
    *,
    capsule_root: Path,
    target_name: str | None,
    all_targets: bool,
    message: str | None,
    preview: bool,
    assume_yes: bool,
    settings: dict[str, Any],
    owner_override: str | None = None,
    repo_override: str | None = None,
    branch_override: str | None = None,
    visibility_override: str | None = None,
    show_review: bool = True,
    show_status: bool = True,
) -> list[dict[str, Any]]:
    github_cfg = settings.get("github", {}) if isinstance(settings.get("github", {}), dict) else {}
    visibility = str(visibility_override or github_cfg.get("default_visibility", "private")).strip().lower() or "private"
    create_repo_if_missing = bool(github_cfg.get("create_repo_if_missing", True))
    default_branch = branch_override or str(github_cfg.get("default_branch", "main")).strip() or "main"
    default_owner = owner_override or str(github_cfg.get("owner", "")).strip() or None

    overridden = _override_target_from_args(
        capsule_root,
        target_name=target_name,
        owner=owner_override,
        repo=repo_override,
        branch=branch_override,
        visibility=visibility,
        preview=preview,
    )
    if str(target_name or "").strip() == "workspace":
        raise SystemExit("`lelabo push` does not support the workspace target. Use `lelabo export` for local exports.")

    targets = _push_targets(capsule_root)
    if overridden is not None:
        targets = _push_targets(capsule_root) if not preview else [overridden, *targets]

    if all_targets:
        if targets:
            chosen = targets
        else:
            chosen = [
                _bootstrap_github_target(
                    capsule_root=capsule_root,
                    owner=default_owner,
                    repo=repo_override,
                    branch=default_branch,
                    visibility=visibility,
                    assume_yes=assume_yes,
                    preview=preview,
                    make_default=True,
                    existing_targets=targets,
                    show_status=show_status,
                )
            ]
    elif target_name:
        if overridden is not None:
            chosen = [overridden]
        else:
            try:
                resolved = resolve_target(capsule_root, target_name)
            except ValueError as exc:
                raise SystemExit(str(exc))
            if str(resolved.get("kind", "")).strip() == "workspace":
                raise SystemExit("`lelabo push` does not support the workspace target. Use `lelabo export` for local exports.")
            chosen = [resolved]
    else:
        default_target_name, last_used_target_name = get_target_preferences(capsule_root)
        default_target = _find_target_by_name(targets, default_target_name or "")
        last_used_target = _find_target_by_name(targets, last_used_target_name or "")
        if overridden is not None:
            chosen = [overridden]
        elif default_target is not None:
            chosen = [default_target]
        elif last_used_target is not None:
            if assume_yes or not _is_interactive_tty():
                chosen = [last_used_target]
            else:
                if _confirm(f"Last used publish target: {_target_summary(last_used_target)}. Push there again?", default=True):
                    chosen = [last_used_target]
                else:
                    chosen = _pick_targets_interactively(
                        capsule_root=capsule_root,
                        targets=targets,
                        owner=default_owner,
                        repo=repo_override,
                        branch=default_branch,
                        visibility=visibility,
                        preview=preview,
                    )
        elif not targets:
            chosen = [
                _bootstrap_github_target(
                    capsule_root=capsule_root,
                    owner=default_owner,
                    repo=repo_override,
                    branch=default_branch,
                    visibility=visibility,
                    assume_yes=assume_yes,
                    preview=preview,
                    make_default=True,
                    existing_targets=targets,
                    show_status=show_status,
                )
            ]
        else:
            if assume_yes and len(targets) == 1:
                chosen = [targets[0]]
            elif _is_interactive_tty():
                chosen = _pick_targets_interactively(
                    capsule_root=capsule_root,
                    targets=targets,
                    owner=default_owner,
                    repo=repo_override,
                    branch=default_branch,
                    visibility=visibility,
                    preview=preview,
                )
            else:
                raise SystemExit(
                    "This capsule has multiple possible publish targets. Run interactively, set a default target, or pass `--target <name>` / `--all-targets`."
                )

    if show_review and _is_interactive_tty():
        _confirm_multiple_targets(chosen, preview=preview)

    results: list[dict[str, Any]] = []
    for target in chosen:
        kind = str(target.get("kind", "")).strip()
        if kind == "github":
            result = push_github_target(
                capsule_root=capsule_root,
                target=target,
                message=message,
                preview=preview,
                create_repo_if_missing=create_repo_if_missing,
            )
        else:
            raise SystemExit(f"Unsupported publish target kind '{kind}'.")
        if bool(target.get("_ephemeral")):
            result["_ephemeral_target"] = True
        results.append(result)
    return results


def _build_parser(*, prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Publish a capsule to configured remote targets.",
        epilog=(
            "Examples:\n"
            f"  {prog} my_capsule\n"
            f"  {prog} my_capsule --target github\n"
            f"  {prog} my_capsule --all-targets --preview"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Optional capsule path or stored id/alias")
    parser.add_argument("--target", default=None, help="Publish target name")
    parser.add_argument("--all-targets", action="store_true", help="Publish to all configured targets")
    parser.add_argument("-m", "--message", default=None, help="Commit message (defaults to `Update <capsule_id>`)")
    parser.add_argument("--preview", action="store_true", help="Show the resolved publish plan without pushing")
    parser.add_argument("--yes", action="store_true", help="Accept default bootstrap answers non-interactively")
    parser.add_argument("--capsules-dir", default=None, help="Override capsules store path for id/alias resolution")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--owner", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repo", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--branch", default=None, help=argparse.SUPPRESS)
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help=argparse.SUPPRESS)
    vis.add_argument("--private", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_push_command(argv: Sequence[str], *, prog: str = "lelabo push") -> tuple[int, dict[str, Any]]:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(list(argv))
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(args.capsules_dir, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir)
    visibility = "public" if bool(args.public) else "private" if bool(args.private) else None
    results = push_capsule(
        capsule_root=capsule_root,
        target_name=args.target,
        all_targets=bool(args.all_targets),
        message=args.message,
        preview=bool(args.preview),
        assume_yes=bool(args.yes),
        settings=settings,
        owner_override=args.owner,
        repo_override=args.repo,
        branch_override=args.branch,
        visibility_override=visibility,
        show_review=not bool(args.json),
        show_status=not bool(args.json),
    )
    payload_results = [{key: value for key, value in row.items() if not str(key).startswith("_")} for row in results]
    payload = {
        "schema_version": PUSH_JSON_SCHEMA if len(payload_results) == 1 else PUSHES_JSON_SCHEMA,
        "command": "push",
        "capsule": {
            "capsule_id": inspect_capsule_directory(capsule_root).get("capsule_id"),
            "path": str(capsule_root),
        },
        "result": payload_results[0] if len(payload_results) == 1 else None,
        "results": payload_results if len(payload_results) > 1 else None,
    }
    if bool(args.json):
        if len(payload_results) == 1:
            payload.pop("results", None)
        else:
            payload.pop("result", None)
        _print_json(payload)
    else:
        if len(results) == 1:
            row = results[0]
            if bool(args.preview) and bool(row.get("_ephemeral_target")):
                print_status("info", "Preview uses an unsaved publish target.")
            if bool(args.preview):
                print_status("info", "Publish preview")
            else:
                print_status("success", "Capsule pushed.")
            print_block(
                "Push",
                (
                    ("target", row.get("target_name")),
                    ("kind", row.get("target_kind")),
                    ("owner", row.get("owner")),
                    ("repo", row.get("repo")),
                    ("branch", row.get("branch")),
                    ("folder", row.get("path")),
                    ("commit_message", row.get("commit_message") or auto_commit_message(capsule_root)),
                    ("committed", row.get("committed")),
                    ("pushed", row.get("pushed")),
                ),
            )
        else:
            header = "Push preview" if bool(args.preview) else "Push results"
            if bool(args.preview) and any(bool(row.get("_ephemeral_target")) for row in results):
                print_status("info", "Preview uses at least one unsaved publish target.")
            if not bool(args.preview):
                print_status("success", f"Pushed {len(results)} targets.")
            print_list_block(
                header,
                [
                    f"{row.get('target_name')} | kind: {row.get('target_kind')} | repo: {row.get('owner')}/{row.get('repo')} | folder: {row.get('path')} | pushed: {row.get('pushed')}"
                    for row in results
                ],
            )
    return 0, payload


def main(argv: Sequence[str]) -> int:
    if argv and str(argv[0]).strip() in {"-h", "--help", "help"}:
        print(PUSH_HELP)
        return 0
    rc, _ = run_push_command(argv)
    return int(rc)


__all__ = [
    "main",
    "push_capsule",
    "run_push_command",
]
