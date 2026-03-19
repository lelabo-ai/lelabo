"""CLI entrypoint for workspace-first sweep discovery and execution."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Sequence

from ...sweep.discovery import DiscoveredCapsuleSweeps, DiscoveredSweep, discover_workspace_capsules, resolve_discovered_sweep
from ...sweep.runner import build_sweep_plan, load_sweep_config, run_sweep
from ..ui import build_prompt_style, print_list_block, print_status


SWEEP_HELP = """\
Run parameter sweeps from the current workspace.

Usage:
  lelabo sweep
  lelabo sweep run [--config PATH | --capsule CAPSULE_ID --sweep SWEEP_NAME] [OPTIONS]

Behavior:
  - `lelabo sweep` discovers capsules under the current workspace.
  - In a TTY, it opens an interactive sweep picker.
  - Outside a TTY, it prints a compact capsule -> sweep listing.

Examples:
  lelabo sweep
  lelabo sweep run --capsule demo_capsule --sweep example
  lelabo sweep run --config sweeps/example.yaml --dry-run
"""


def _is_interactive_tty() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _command_preview(
    *,
    config_path: Path,
    outdir: str,
    name: str | None,
    max_parallel: int,
    gpus: str | None,
    dry_run: bool,
) -> list[str]:
    cmd = ["lelabo", "sweep", "run", "--config", str(config_path)]
    if outdir != "outputs/runs":
        cmd.extend(["--outdir", str(outdir)])
    if name:
        cmd.extend(["--name", str(name)])
    if max_parallel != 1:
        cmd.extend(["--max-parallel", str(max_parallel)])
    if gpus:
        cmd.extend(["--gpus", str(gpus)])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _run_sweep_from_config(
    *,
    config_path: Path,
    outdir: str,
    name: str | None,
    max_parallel: int,
    gpus: str | None,
    dry_run: bool,
) -> int:
    cfg = load_sweep_config(config_path)
    plan = build_sweep_plan(
        config=cfg,
        outdir=Path(outdir),
        name=name,
        gpus=gpus,
        wandb_group=name or cfg.get("name"),
    )
    result = run_sweep(plan, max_parallel=max_parallel, dry_run=dry_run)
    return 1 if result.get("failed", 0) > 0 and not result.get("dry_run") else 0


def _print_workspace_listing(capsules: tuple[DiscoveredCapsuleSweeps, ...]) -> int:
    non_empty = [capsule for capsule in capsules if capsule.sweeps]
    if not non_empty:
        print_status(
            "info",
            "No sweeps found in the current workspace. Create a capsule or run `lelabo sweep run --config path/to/sweep.yaml`.",
        )
        return 0

    for idx, capsule in enumerate(non_empty):
        if idx:
            print("")
        print_list_block(
            capsule.capsule_id,
            [f"{item.sweep_name} | path: {item.config_path.relative_to(capsule.capsule_root)}" for item in capsule.sweeps],
        )
    return 0


def _pick_sweep_interactive(capsules: tuple[DiscoveredCapsuleSweeps, ...]) -> DiscoveredSweep | None:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.widgets import Label
    except Exception as exc:
        raise RuntimeError(
            "`prompt_toolkit` is required for interactive sweep selection. "
            "Install dependencies or run `lelabo sweep run --config ...`."
        ) from exc

    lines: list[tuple[str, DiscoveredSweep | None]] = []
    for capsule in capsules:
        if not capsule.sweeps:
            continue
        lines.append((capsule.capsule_id, None))
        for sweep in capsule.sweeps:
            lines.append((f"  {sweep.sweep_name}", sweep))
    if not lines:
        return None

    cursor = 0
    state = {"error": ""}

    def _move(step: int) -> None:
        nonlocal cursor
        cursor = max(0, min(len(lines) - 1, cursor + step))

    def _render_text():
        fragments: list[tuple[str, str]] = []
        for idx, (label, sweep) in enumerate(lines):
            prefix = "› " if idx == cursor else "  "
            if sweep is None:
                style = "class:accent"
            else:
                style = "class:checkbox-list.current" if idx == cursor else "class:checkbox-list"
            fragments.append((style, f"{prefix}{label}\n"))
        if state["error"]:
            fragments.append(("class:error", f"\n{state['error']}\n"))
        return fragments

    body = Window(
        content=FormattedTextControl(_render_text),
        always_hide_cursor=True,
    )

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:
        _move(-1)
        state["error"] = ""
        event.app.invalidate()

    @kb.add("down")
    def _down(event) -> None:
        _move(1)
        state["error"] = ""
        event.app.invalidate()

    @kb.add("enter", eager=True)
    def _enter(event) -> None:
        _, selected = lines[cursor]
        if selected is None:
            state["error"] = "Select a sweep, not a capsule header."
            event.app.invalidate()
            return
        event.app.exit(result=selected)

    @kb.add("escape", eager=True)
    @kb.add("c-c", eager=True)
    def _cancel(event) -> None:
        event.app.exit(result=None)

    root = HSplit(
        [
            Label(text="Workspace sweeps", style="class:accent"),
            Label(text="Select a sweep and press Enter for a preview.", style="class:muted"),
            body,
            Label(text="Keys: Up/Down move | Enter preview | Esc cancel", style="class:muted"),
        ],
        padding=1,
    )
    app: Application[DiscoveredSweep | None] = Application(
        layout=Layout(root),
        key_bindings=kb,
        mouse_support=True,
        full_screen=True,
        style=build_prompt_style(),
    )
    return app.run()


def _preview_and_confirm_run(
    sweep: DiscoveredSweep,
    *,
    outdir: str,
    name: str | None,
    max_parallel: int,
    gpus: str | None,
    dry_run: bool,
) -> tuple[bool, list[str]]:
    if not _is_interactive_tty():
        return True, []

    extra_flags: list[str] = []
    current_outdir = outdir
    current_name = name
    current_parallel = max_parallel
    current_gpus = gpus
    current_dry_run = dry_run

    try:
        from prompt_toolkit.shortcuts import button_dialog, input_dialog
    except Exception:
        return True, []

    while True:
        cmd_preview = _command_preview(
            config_path=sweep.config_path,
            outdir=current_outdir,
            name=current_name,
            max_parallel=current_parallel,
            gpus=current_gpus,
            dry_run=current_dry_run,
        )
        result = button_dialog(
            title="Sweep preview",
            text=(
                f"capsule: {sweep.capsule_id}\n"
                f"sweep: {sweep.sweep_name}\n"
                f"config_path: {sweep.config_path}\n"
                f"resolved_command: {shlex.join(cmd_preview)}\n"
                f"overrides: {shlex.join(extra_flags) if extra_flags else '-'}"
            ),
            buttons=[
                ("Run", "run"),
                ("Edit flags", "edit"),
                ("Cancel", "cancel"),
            ],
            style=build_prompt_style(),
        ).run()
        if result == "run":
            return True, extra_flags
        if result == "cancel" or result is None:
            return False, []
        raw = input_dialog(
            title="Edit flags",
            text="Add optional sweep flags, e.g. --dry-run --max-parallel 4 --gpus 0,1",
            default=shlex.join(extra_flags),
            style=build_prompt_style(),
        ).run()
        if raw is None:
            continue
        extra_flags = shlex.split(str(raw).strip()) if str(raw).strip() else []
        parser = _build_run_parser()
        parsed = parser.parse_args(["--config", str(sweep.config_path), *extra_flags])
        current_outdir = str(parsed.outdir)
        current_name = parsed.name
        current_parallel = int(parsed.max_parallel)
        current_gpus = parsed.gpus
        current_dry_run = bool(parsed.dry_run)


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lelabo sweep run",
        description="Run a named sweep from the current workspace, or execute a sweep config directly.",
        epilog=(
            "Examples:\n"
            "  lelabo sweep run --capsule demo_capsule --sweep example\n"
            "  lelabo sweep run --sweep example --dry-run\n"
            "  lelabo sweep run --config sweeps/example.yaml --max-parallel 4 --gpus 0,1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Direct path to a sweep YAML config.")
    parser.add_argument("--capsule", type=str, default=None, help="Capsule id resolved from the current workspace.")
    parser.add_argument("--sweep", type=str, default=None, help="Named sweep from `sweeps/<name>.yaml`.")
    parser.add_argument("--outdir", type=str, default="outputs/runs", help="Where logs and metadata are written.")
    parser.add_argument("--name", type=str, default=None, help="Override the sweep run name.")
    parser.add_argument("--max-parallel", type=int, default=1, help="Number of concurrent runs.")
    parser.add_argument("--gpus", type=str, default=None, help="GPU ids for round-robin, e.g. `0,1`.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def _resolve_run_target(parsed: argparse.Namespace) -> tuple[Path, DiscoveredSweep | None]:
    if parsed.config:
        return Path(str(parsed.config)).expanduser().resolve(), None

    capsules = discover_workspace_capsules()
    non_empty = tuple(capsule for capsule in capsules if capsule.sweeps)
    if not non_empty:
        raise SystemExit(
            "No sweeps found in the current workspace. "
            "Use `lelabo sweep run --config path/to/sweep.yaml`."
        )

    if parsed.sweep:
        try:
            resolved = resolve_discovered_sweep(non_empty, capsule_id=parsed.capsule, sweep_name=parsed.sweep)
        except ValueError as exc:
            raise SystemExit(str(exc))
        return resolved.config_path, resolved

    if _is_interactive_tty():
        selected = _pick_sweep_interactive(non_empty)
        if selected is None:
            raise SystemExit("Sweep selection canceled by user.")
        return selected.config_path, selected

    raise SystemExit(
        "Sweep selection is ambiguous in non-interactive mode. "
        "Use `lelabo sweep run --config <path>` or `lelabo sweep run --capsule <capsule_id> --sweep <name>`."
    )


def _cmd_run(argv: list[str]) -> int:
    parser = _build_run_parser()
    parsed = parser.parse_args(argv)
    config_path, selected = _resolve_run_target(parsed)
    should_run = True
    extra_flags: list[str] = []
    if selected is not None:
        should_run, extra_flags = _preview_and_confirm_run(
            selected,
            outdir=str(parsed.outdir),
            name=parsed.name,
            max_parallel=int(parsed.max_parallel),
            gpus=parsed.gpus,
            dry_run=bool(parsed.dry_run),
        )
        if not should_run:
            raise SystemExit("Sweep launch canceled by user.")
        if extra_flags:
            parsed = parser.parse_args(["--config", str(config_path), *extra_flags])

    return _run_sweep_from_config(
        config_path=config_path,
        outdir=str(parsed.outdir),
        name=parsed.name,
        max_parallel=int(parsed.max_parallel),
        gpus=parsed.gpus,
        dry_run=bool(parsed.dry_run),
    )


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if args and args[0] in {"-h", "--help", "help"}:
        print(SWEEP_HELP)
        return 0

    if not args:
        capsules = discover_workspace_capsules()
        return _pick_or_list_workspace_sweeps(capsules)

    cmd = str(args[0]).strip().lower()
    rest = args[1:]
    if cmd == "run":
        return _cmd_run(rest)

    raise SystemExit(
        f"Unknown sweep subcommand: {cmd}\n\n"
        "Use `lelabo sweep` for the workspace dashboard or `lelabo sweep run ...` to execute a sweep."
    )


def _pick_or_list_workspace_sweeps(capsules: tuple[DiscoveredCapsuleSweeps, ...]) -> int:
    non_empty = tuple(capsule for capsule in capsules if capsule.sweeps)
    if not non_empty:
        print_status(
            "info",
            "No sweeps found in the current workspace. Create a capsule or run `lelabo sweep run --config path/to/sweep.yaml`.",
        )
        return 0

    if not _is_interactive_tty():
        return _print_workspace_listing(non_empty)

    selected = _pick_sweep_interactive(non_empty)
    if selected is None:
        print_status("info", "Sweep selection canceled.")
        return 0
    should_run, extra_flags = _preview_and_confirm_run(
        selected,
        outdir="outputs/runs",
        name=None,
        max_parallel=1,
        gpus=None,
        dry_run=False,
    )
    if not should_run:
        print_status("info", "Sweep launch canceled.")
        return 0
    run_args = ["--config", str(selected.config_path), *extra_flags]
    return _cmd_run(run_args)


__all__ = ["main"]
