"""Capsule-first publish command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ...capsule import get_capsule, inspect_capsule_directory
from ...capsule.plugins.discovery import find_active_capsule_root
from ...capsule.publish import (
    auto_commit_message,
    available_targets,
    current_github_login,
    infer_workspace_target,
    list_configured_targets,
    parse_owner_repo_from_origin,
    push_github_target,
    push_workspace_target,
    resolve_target,
    save_configured_target,
)
from ...config.user_settings import load_effective_settings
from ..ui import print_block, print_list_block, print_status


PUSH_JSON_SCHEMA = "lelabo.cli.push/v1"
PUSHES_JSON_SCHEMA = "lelabo.cli.pushes/v1"


PUSH_HELP = """\
Publish one capsule to a workspace repo or configured remote target.

Usage:
  lelabo push [capsule_ref] [--target NAME] [--all-targets] [-m MESSAGE] [--preview] [--yes] [--json]

Notes:
  - `capsule_ref` can be a local path or a stored capsule id/alias
  - if a capsule already lives in a git repo with `origin`, LeLabo uses the implicit `workspace` target
  - otherwise LeLabo can bootstrap a GitHub target and remember it for next time
"""


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _is_capsule_root(path: Path) -> bool:
    return path.is_dir() and ((path / "capsule.toml").is_file() or (path / "manifest.json").is_file())


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
    answer = input(f"{label} [{token}]: ").strip()
    return answer or token


def _resolve_capsule_root(capsule_ref: str | None, *, caps_dir: Path | None) -> Path:
    if not capsule_ref:
        active = find_active_capsule_root()
        if active is not None:
            return active.resolve()
        direct_children = sorted(child.resolve() for child in Path.cwd().iterdir() if _is_capsule_root(child))
        if len(direct_children) == 1:
            return direct_children[0]
        raise SystemExit("No active capsule found. Pass a capsule path/id or run inside a capsule directory.")

    ref_path = Path(capsule_ref).expanduser()
    if ref_path.exists():
        start = ref_path.resolve()
        if start.is_file():
            start = start.parent
        root = find_active_capsule_root(start=start)
        if root is None and _is_capsule_root(start):
            root = start
        if root is None:
            raise SystemExit(f"Path '{capsule_ref}' is not inside a capsule (missing capsule.toml).")
        return root.resolve()

    local_candidate = (Path.cwd() / str(capsule_ref)).resolve()
    if _is_capsule_root(local_candidate):
        return local_candidate

    row = get_capsule(capsule_ref, caps_dir)
    if row is None:
        raise SystemExit(f"Unknown capsule '{capsule_ref}' (not found as path nor stored id/alias).")
    root = Path(str(row.get("path", ""))).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Capsule path does not exist on disk: {root}")
    return root


def _pick_target_interactively(targets: list[dict[str, Any]]) -> dict[str, Any]:
    print_list_block(
        "Publish targets",
        [
            f"{idx}. {item['name']} | kind: {item.get('kind')} | repo: {item.get('owner', '-')}/{item.get('repo', '-')} | branch: {item.get('branch', '-')}"
            for idx, item in enumerate(targets, start=1)
        ],
    )
    while True:
        answer = input("Target number (blank to cancel): ").strip()
        if not answer:
            raise SystemExit("Push canceled by user.")
        if answer.isdigit():
            idx = int(answer)
            if 1 <= idx <= len(targets):
                return targets[idx - 1]
        print_status("warning", "Enter a valid target number.")


def _bootstrap_github_target(
    *,
    capsule_root: Path,
    owner: str | None,
    repo: str | None,
    branch: str | None,
    visibility: str,
    assume_yes: bool,
) -> dict[str, Any]:
    capsule_id = str(inspect_capsule_directory(capsule_root).get("capsule_id", capsule_root.name)).strip() or capsule_root.name
    login = current_github_login()
    default_owner = str(owner or "").strip() or str(login or "").strip()
    default_repo = str(repo or "").strip() or capsule_id
    default_branch = str(branch or "").strip() or "main"
    default_path = capsule_id
    default_name = "github"
    resolved_visibility = str(visibility).strip().lower() or "private"

    if not assume_yes:
        if not _is_interactive_tty():
            raise SystemExit(
                "No publish target is configured for this capsule. "
                "Run interactively to bootstrap one, or pass `--yes` with enough GitHub defaults configured."
            )
        print_status("info", "No publish target is configured for this capsule yet.")
        print_block(
            "GitHub account",
            (
                ("detected_login", login or "-"),
                ("capsule_id", capsule_id),
            ),
        )
        if not _confirm("Create a GitHub publish target now?", default=True):
            raise SystemExit("Publish target bootstrap canceled by user.")
        default_name = _prompt_with_default("Target name", default_name)
        default_owner = _prompt_with_default("GitHub owner", default_owner or "")
        default_repo = _prompt_with_default("GitHub repo", default_repo)
        default_branch = _prompt_with_default("Branch", default_branch)
        default_path = _prompt_with_default("Path in repo", default_path)
        resolved_visibility = _prompt_with_default("Visibility (private/public)", resolved_visibility)
        if resolved_visibility not in {"public", "private"}:
            raise SystemExit("Visibility must be 'public' or 'private'.")
        print_block(
            "Review",
            (
                ("target", default_name),
                ("owner", default_owner),
                ("repo", default_repo),
                ("branch", default_branch),
                ("path", default_path),
                ("visibility", resolved_visibility),
            ),
        )
        if not _confirm("Continue?", default=True):
            raise SystemExit("Publish target bootstrap canceled by user.")
    else:
        if not default_owner:
            raise SystemExit(
                "GitHub owner could not be resolved. Set --owner or `lelabo config set github.owner <owner>`."
            )

    target = save_configured_target(
        capsule_root,
        {
            "name": default_name,
            "kind": "github",
            "owner": default_owner,
            "repo": default_repo,
            "branch": default_branch,
            "path": default_path,
            "visibility": resolved_visibility,
        },
        make_default=True,
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
    return save_configured_target(
        capsule_root,
        {
            "name": name,
            "kind": "github",
            "owner": str(owner or current.get("owner") or "").strip(),
            "repo": str(repo or current.get("repo") or capsule_id).strip(),
            "branch": str(branch or current.get("branch") or "main").strip() or "main",
            "path": str(current.get("path") or capsule_id).strip() or capsule_id,
            "visibility": str(visibility or current.get("visibility") or "private").strip().lower() or "private",
        },
        make_default=True,
    )


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
) -> list[dict[str, Any]]:
    github_cfg = settings.get("github", {}) if isinstance(settings.get("github", {}), dict) else {}
    visibility = str(visibility_override or github_cfg.get("default_visibility", "private")).strip().lower() or "private"
    create_repo_if_missing = bool(github_cfg.get("create_repo_if_missing", True))

    overridden = _override_target_from_args(
        capsule_root,
        target_name=target_name,
        owner=owner_override,
        repo=repo_override,
        branch=branch_override,
        visibility=visibility,
    )
    targets = available_targets(capsule_root)
    if overridden is not None:
        targets = available_targets(capsule_root)

    if all_targets:
        chosen = targets
    elif target_name:
        try:
            chosen = [resolve_target(capsule_root, target_name)]
        except ValueError as exc:
            raise SystemExit(str(exc))
    else:
        preferred = None
        if overridden is not None:
            preferred = overridden
        else:
            if len(targets) == 1:
                preferred = targets[0]
            else:
                for item in targets:
                    if bool(item.get("default")):
                        preferred = item
                        break
                if preferred is None:
                    for item in targets:
                        if bool(item.get("last_used")):
                            preferred = item
                            break
        if preferred is not None:
            chosen = [preferred]
        else:
            workspace = infer_workspace_target(capsule_root)
            if workspace is not None and not list_configured_targets(capsule_root):
                chosen = [workspace]
            else:
                owner = owner_override or str(github_cfg.get("owner", "")).strip() or None
                repo = repo_override
                branch = branch_override or str(github_cfg.get("default_branch", "main")).strip() or "main"
                if not targets:
                    chosen = [_bootstrap_github_target(
                        capsule_root=capsule_root,
                        owner=owner,
                        repo=repo,
                        branch=branch,
                        visibility=visibility,
                        assume_yes=assume_yes,
                    )]
                elif _is_interactive_tty():
                    chosen = [_pick_target_interactively(targets)]
                else:
                    raise SystemExit(
                        "This capsule has multiple publish targets. Use `--target <name>` or `--all-targets`."
                    )

    results: list[dict[str, Any]] = []
    for target in chosen:
        kind = str(target.get("kind", "")).strip()
        if kind == "workspace":
            result = push_workspace_target(
                capsule_root=capsule_root,
                target=target,
                message=message,
                preview=preview,
            )
        elif kind == "github":
            result = push_github_target(
                capsule_root=capsule_root,
                target=target,
                message=message,
                preview=preview,
                create_repo_if_missing=create_repo_if_missing,
            )
        else:
            raise SystemExit(f"Unsupported publish target kind '{kind}'.")
        results.append(result)
    return results


def _build_parser(*, prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Publish a capsule to a workspace repo or configured remote target.",
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
    )
    payload = {
        "schema_version": PUSH_JSON_SCHEMA if len(results) == 1 else PUSHES_JSON_SCHEMA,
        "command": "push",
        "capsule": {
            "capsule_id": inspect_capsule_directory(capsule_root).get("capsule_id"),
            "path": str(capsule_root),
        },
        "result": results[0] if len(results) == 1 else None,
        "results": results if len(results) > 1 else None,
    }
    if bool(args.json):
        if len(results) == 1:
            payload.pop("results", None)
        else:
            payload.pop("result", None)
        _print_json(payload)
    else:
        if len(results) == 1:
            row = results[0]
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
                    ("path", row.get("path")),
                    ("commit_message", row.get("commit_message") or auto_commit_message(capsule_root)),
                    ("committed", row.get("committed")),
                    ("pushed", row.get("pushed")),
                ),
            )
        else:
            header = "Push preview" if bool(args.preview) else "Push results"
            if not bool(args.preview):
                print_status("success", f"Pushed {len(results)} targets.")
            print_list_block(
                header,
                [
                    f"{row.get('target_name')} | kind: {row.get('target_kind')} | repo: {row.get('owner')}/{row.get('repo')} | pushed: {row.get('pushed')}"
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
