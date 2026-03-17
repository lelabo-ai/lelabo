"""CLI entrypoint for sweep operations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


SWEEP_HELP = """\
Run parameter sweep experiments.

Usage:
  lelabo sweep <subcommand> [args]

Subcommands:
  run        Execute a grid sweep from a YAML config

Help:
  lelabo sweep -h
  lelabo sweep run -h

Examples:
  lelabo sweep run --config experiments/sweeps/demo.yaml
  lelabo sweep run --config sweep.yaml --max-parallel 4 --gpus 0,1
  lelabo sweep run --config sweep.yaml --dry-run
"""


def _cmd_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="lelabo sweep run",
        description="Execute a grid sweep from a YAML config file.",
        epilog=(
            "Examples:\n"
            "  lelabo sweep run --config experiments/sweeps/demo.yaml\n"
            "  lelabo sweep run --config sweep.yaml --max-parallel 4 --gpus 0,1\n"
            "  lelabo sweep run --config sweep.yaml --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config describing base args + grid.")
    parser.add_argument("--outdir", type=str, default="outputs/runs", help="Where logs/metadata are written")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (default: from config or timestamp)")
    parser.add_argument("--max-parallel", type=int, default=1, help="Number of concurrent runs")
    parser.add_argument("--gpus", type=str, default=None, help="GPU ids for round-robin, e.g. '0,1,2'")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args(argv)

    from ...sweep.runner import build_sweep_plan, load_sweep_config, run_sweep

    cfg = load_sweep_config(Path(args.config))
    plan = build_sweep_plan(
        config=cfg,
        outdir=Path(args.outdir),
        name=args.name,
        gpus=args.gpus,
        wandb_group=args.name or cfg.get("name"),
    )
    result = run_sweep(plan, max_parallel=args.max_parallel, dry_run=args.dry_run)
    return 1 if result.get("failed", 0) > 0 and not result.get("dry_run") else 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(SWEEP_HELP)
        return 0

    cmd = args[0]
    rest = args[1:]

    if cmd == "run":
        return _cmd_run(rest)

    raise SystemExit(
        f"Unknown sweep subcommand: {cmd}\n\n"
        "Use one of: run.\n"
        "Run `lelabo sweep -h` for usage."
    )
