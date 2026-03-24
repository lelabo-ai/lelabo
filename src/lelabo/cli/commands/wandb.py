"""CLI helpers for Weights & Biases project setup."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from ...config.user_settings import find_local_override, load_settings_file, read_settings_file, set_key, write_settings
from ..ui import print_block, print_status


WANDB_HELP = """\
Manage Weights & Biases setup for LeLabo projects.

Usage:
  lelabo wandb <subcommand> [args]

Subcommands:
  init      Initialize W&B config for the current project
  login     Authenticate with W&B on this machine
  status    Show machine-level and project-level W&B status
"""


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


def _wandb_package_available() -> bool:
    try:
        return importlib.util.find_spec("wandb") is not None
    except Exception:
        return False


def _wandb_cli_available() -> bool:
    return bool(shutil.which("wandb"))


def _wandb_login_state() -> tuple[bool, str]:
    api_key = str(os.getenv("WANDB_API_KEY", "")).strip()
    if api_key:
        return True, "env:WANDB_API_KEY"
    netrc_path = Path.home() / ".netrc"
    if netrc_path.is_file():
        try:
            content = netrc_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            content = ""
        if "api.wandb.ai" in content:
            return True, str(netrc_path)
    return False, "-"


def _project_config_path() -> Path:
    existing = find_local_override(Path.cwd())
    if existing is not None:
        return existing
    return (Path.cwd() / ".lelabo" / "config.toml").resolve()


def _current_project_wandb_config() -> tuple[Path | None, dict[str, Any] | None]:
    local_path = find_local_override(Path.cwd())
    if local_path is None:
        return None, None
    raw = read_settings_file(local_path)
    wandb_raw = raw.get("wandb", {})
    if not isinstance(wandb_raw, dict) or not wandb_raw:
        return local_path, None
    merged = load_settings_file(local_path)
    wandb_cfg = merged.get("wandb", {})
    return local_path, dict(wandb_cfg) if isinstance(wandb_cfg, dict) else None


def _run_wandb_login() -> None:
    if not _wandb_cli_available():
        raise SystemExit("W&B CLI is not installed. Install it with `pip install wandb`, then run `wandb login`.")
    proc = subprocess.run(["wandb", "login"], check=False)
    if proc.returncode != 0:
        raise SystemExit("`wandb login` failed.")


def _print_status(*, include_hint: bool = True) -> None:
    local_path, local_cfg = _current_project_wandb_config()
    logged_in, login_source = _wandb_login_state()
    configured = isinstance(local_cfg, dict)
    print_block(
        "W&B status",
        (
            ("package", "installed" if _wandb_package_available() else "missing"),
            ("cli", "available" if _wandb_cli_available() else "missing"),
            ("logged_in", logged_in),
            ("login_source", login_source),
            ("project_config", local_path if local_path is not None else "-"),
            ("configured", configured),
            ("project", (local_cfg or {}).get("project") if configured else "-"),
            ("entity", (local_cfg or {}).get("entity") if configured else "-"),
            ("enabled", (local_cfg or {}).get("enabled") if configured else "-"),
        ),
    )
    if include_hint and not configured:
        print_status("info", "No local W&B project config was found. Run `lelabo wandb init` in this workspace.")


def _cmd_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo wandb status", description="Show W&B machine and project status.")
    args = parser.parse_args(argv)
    _ = args
    _print_status()
    return 0


def _cmd_login(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lelabo wandb login", description="Authenticate with W&B on this machine.")
    args = parser.parse_args(argv)
    _ = args
    _run_wandb_login()
    print_status("success", "W&B login completed.")
    local_path, local_cfg = _current_project_wandb_config()
    if local_cfg is None:
        print_status("info", "Run `lelabo wandb init` to configure this project.")
    elif local_path is not None:
        print_block("Project", (("config", local_path), ("project", local_cfg.get("project")), ("entity", local_cfg.get("entity"))))
    return 0


def _cmd_init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo wandb init",
        description="Initialize W&B config for the current LeLabo project.",
    )
    parser.add_argument("--project", default=None, help="W&B project name for this workspace")
    parser.add_argument("--entity", default=None, help="Optional W&B entity/team for this workspace")
    enabled = parser.add_mutually_exclusive_group()
    enabled.add_argument("--enable", action="store_true", help="Enable W&B by default for this workspace")
    enabled.add_argument("--disable", action="store_true", help="Disable W&B by default for this workspace")
    args = parser.parse_args(argv)

    target = _project_config_path()
    existing_raw = read_settings_file(target)
    existing_raw_wandb = existing_raw.get("wandb", {}) if isinstance(existing_raw.get("wandb", {}), dict) else {}
    existing = load_settings_file(target)
    existing_wandb = existing.get("wandb", {}) if isinstance(existing.get("wandb", {}), dict) else {}
    if existing_raw_wandb:
        print_block(
            "Current W&B config",
            (
                ("file", target),
                ("project", existing_wandb.get("project")),
                ("entity", existing_wandb.get("entity")),
                ("enabled", existing_wandb.get("enabled")),
            ),
        )

    project = str(args.project or "").strip()
    entity = str(args.entity or "").strip()
    if args.enable:
        enabled_value = True
    elif args.disable:
        enabled_value = False
    else:
        enabled_value = bool(existing_wandb.get("enabled", True))

    if not project:
        if not _is_interactive_tty():
            raise SystemExit("Missing `--project`. Use `lelabo wandb init --project <name>` or run interactively.")
        project = _prompt_with_default("W&B project", str(existing_wandb.get("project", "") or ""))
    if not project:
        raise SystemExit("W&B project is required.")

    if not entity and _is_interactive_tty():
        entity = _prompt_with_default("W&B entity", str(existing_wandb.get("entity", "") or os.getenv("WANDB_ENTITY", "")))
    if not args.enable and not args.disable and _is_interactive_tty():
        enabled_value = _confirm("Enable W&B by default for this workspace?", default=enabled_value)

    updated = set_key(existing, "wandb.project", project)
    updated = set_key(updated, "wandb.entity", entity)
    updated = set_key(updated, "wandb.enabled", "true" if enabled_value else "false")
    write_settings(target, updated)

    print_status("success", "W&B project config initialized.")
    print_block(
        "W&B config",
        (
            ("file", target),
            ("project", project),
            ("entity", entity or "-"),
            ("enabled", enabled_value),
        ),
    )

    logged_in, _ = _wandb_login_state()
    if not logged_in:
        if _wandb_cli_available():
            if _is_interactive_tty() and _confirm("W&B login is not configured. Run `wandb login` now?", default=True):
                _run_wandb_login()
                print_status("success", "W&B login completed.")
            else:
                print_status("info", "Run `lelabo wandb login` when you are ready to authenticate.")
        else:
            print_status("info", "W&B CLI is not installed. Install it with `pip install wandb`, then run `lelabo wandb login`.")
    return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(WANDB_HELP)
        return 0
    cmd = str(args[0]).strip().lower()
    rest = args[1:]
    if cmd == "init":
        return _cmd_init(rest)
    if cmd == "login":
        return _cmd_login(rest)
    if cmd == "status":
        return _cmd_status(rest)
    raise SystemExit(
        f"Unknown wandb subcommand: {cmd}\n\n"
        "Use one of: init, login, status.\n"
        "Run `lelabo wandb -h` for usage."
    )


__all__ = ["main"]
