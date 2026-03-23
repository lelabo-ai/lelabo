"""Capsule-first publish command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ...capsule import inspect_capsule_directory
from ...capsule.discovery import resolve_visible_capsule_ref
from ...capsule.publish import (
    auto_commit_message,
    available_targets,
    get_target_preferences,
    push_github_target,
    resolve_target,
)
from ...config.user_settings import load_effective_settings
from ..interactive_picker import pick_many_with_checkboxes
from ..ui import print_block, print_list_block, print_status


PUSH_JSON_SCHEMA = "lelabo.cli.push/v1"
PUSHES_JSON_SCHEMA = "lelabo.cli.pushes/v1"
PUSH_HELP = """\
Publish one capsule to configured remote targets.

Usage:
  lelabo push <capsule> [--target NAME] [--all-targets] [-m MESSAGE] [--preview] [--yes] [--json]

Notes:
  - pass a capsule id or alias from `lelabo capsule list`
  - `lelabo push` publishes to saved remote targets
  - local filesystem paths are not accepted by `lelabo push` in v1
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


def _resolve_capsule_root(capsule_ref: str | None, *, caps_dir: Path | None) -> Path:
    try:
        return resolve_visible_capsule_ref(
            capsule_ref,
            start=Path.cwd(),
            capsules_dir=caps_dir,
            command="lelabo push",
            usage="lelabo push <capsule>",
        ).root
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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


def _pick_targets_interactively(
    *,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = pick_many_with_checkboxes(
        title="Select publish targets",
        text="Choose one or more remote publish targets.",
        options=[(str(item.get("name", "")).strip(), _target_picker_label(item)) for item in targets],
        empty_selection_message="Select at least one publish target before confirming.",
        selection_noun="target",
        confirm_button_text="Confirm selected targets",
    )
    if selected is None:
        raise SystemExit("Push canceled by user.")
    chosen = [item for item in targets if str(item.get("name", "")).strip() in {str(token).strip() for token in selected}]
    if not chosen:
        raise SystemExit("Push canceled by user.")
    return chosen


def _confirm_multiple_targets(
    chosen: Sequence[dict[str, Any]],
    *,
    preview: bool,
    question: str | None = None,
    cancel_message: str | None = None,
) -> None:
    if len(chosen) <= 1:
        return
    print_list_block("Publish targets", [_target_summary(item) for item in chosen])
    prompt = str(question or ("Continue with this preview?" if preview else "Continue?")).strip()
    if not _confirm(prompt, default=True):
        raise SystemExit(cancel_message or "Push canceled by user.")


def push_capsule(
    *,
    capsule_root: Path,
    target_name: str | None,
    all_targets: bool,
    message: str | None,
    preview: bool,
    assume_yes: bool,
    settings: dict[str, Any],
    show_review: bool = True,
) -> list[dict[str, Any]]:
    github_cfg = settings.get("github", {}) if isinstance(settings.get("github", {}), dict) else {}
    create_repo_if_missing = bool(github_cfg.get("create_repo_if_missing", True))
    if str(target_name or "").strip() == "workspace":
        raise SystemExit("`lelabo push` does not support the workspace target. Use `lelabo export` for local exports.")

    targets = _push_targets(capsule_root)

    if all_targets:
        if not targets:
            raise SystemExit(
                "No publish target is configured for this capsule. "
                "Create one with `lelabo targets create`, then attach this capsule with `lelabo targets edit <owner/repo>`."
            )
        chosen = targets
    elif target_name:
        try:
            resolved = resolve_target(capsule_root, target_name)
        except ValueError as exc:
            raise SystemExit(str(exc))
        if str(resolved.get("kind", "")).strip() == "workspace":
            raise SystemExit("`lelabo push` does not support the workspace target. Use `lelabo export` for local exports.")
        chosen = [resolved]
    else:
        default_target_names, last_used_target_name = get_target_preferences(capsule_root)
        default_targets = [item for item in targets if str(item.get("name", "")).strip() in set(default_target_names)]
        last_used_target = _find_target_by_name(targets, last_used_target_name or "")
        if default_targets:
            chosen = default_targets
        elif last_used_target is not None:
            if assume_yes or not _is_interactive_tty():
                chosen = [last_used_target]
            else:
                if _confirm(f"Last used publish target: {_target_summary(last_used_target)}. Push there again?", default=True):
                    chosen = [last_used_target]
                else:
                    chosen = _pick_targets_interactively(targets=targets)
        elif not targets:
            raise SystemExit(
                "No publish target is configured for this capsule. "
                "Create one with `lelabo targets create`, then attach this capsule with `lelabo targets edit <owner/repo>`."
            )
        else:
            if assume_yes and len(targets) == 1:
                chosen = [targets[0]]
            elif _is_interactive_tty():
                chosen = _pick_targets_interactively(targets=targets)
            else:
                raise SystemExit(
                    "This capsule has multiple possible publish targets. Run interactively, set default targets, or pass `--target <name>` / `--all-targets`."
                )

    if show_review and _is_interactive_tty():
        cancel_message = None
        question = None
        if len(chosen) > 1:
            chosen_names = {str(item.get("name", "")).strip() for item in chosen}
            default_names = set(default_target_names) if 'default_target_names' in locals() else set()
            if chosen_names and chosen_names == default_names:
                question = "Push to all default targets?"
                cancel_message = (
                    "Push canceled. Use `--target <name>` to push to one target, or change the default targets for this capsule."
                )
        _confirm_multiple_targets(chosen, preview=preview, question=question, cancel_message=cancel_message)

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
    parser.add_argument("capsule_ref", nargs="?", default=None, help="Capsule id or alias")
    parser.add_argument("--target", default=None, help="Publish target name")
    parser.add_argument("--all-targets", action="store_true", help="Publish to all configured targets")
    parser.add_argument("-m", "--message", default=None, help="Commit message (defaults to `Update <capsule_id>`)")
    parser.add_argument("--preview", action="store_true", help="Show the resolved publish plan without pushing")
    parser.add_argument("--yes", action="store_true", help="Accept confirmation prompts non-interactively")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    return parser


def run_push_command(argv: Sequence[str], *, prog: str = "lelabo push") -> tuple[int, dict[str, Any]]:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(list(argv))
    if not str(args.capsule_ref or "").strip():
        raise SystemExit("Missing capsule. Use `lelabo push <capsule>`. Run `lelabo capsule list` to inspect available capsules.")
    settings = _effective_settings()
    caps_dir = _resolved_capsules_dir(None, settings)
    capsule_root = _resolve_capsule_root(args.capsule_ref, caps_dir=caps_dir)
    results = push_capsule(
        capsule_root=capsule_root,
        target_name=args.target,
        all_targets=bool(args.all_targets),
        message=args.message,
        preview=bool(args.preview),
        assume_yes=bool(args.yes),
        settings=settings,
        show_review=not bool(args.json),
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
