"""Static checks and runtime audits for update rule compatibility."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence


def build_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lelabo audit",
        description="Run warn-only local update-rule audit.",
    )
    parser.add_argument(
        "rule",
        nargs="?",
        help="Update rule to audit (e.g. bp, fa, dni). Ignored with --all.",
    )
    parser.add_argument("--all", action="store_true", help="Audit all registered update rules.")
    parser.add_argument(
        "--modes",
        default="supervised,rl",
        help="Comma-separated modes to audit. Default: supervised,rl",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Supervised model(s) for the audit. "
            "Use auto/all (default set), one model (mlp/cnn/transformer/deephebb), "
            "or a comma-separated list."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of audit epochs to run. Default: 1")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=1,
        help="Number of train_step calls per epoch. Default: 1",
    )
    parser.add_argument("--no-quiet", action="store_true", help="Disable pytest -q to show full output.")
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show all warnings during audit execution.",
    )
    return parser


def run_audit(argv: Sequence[str]) -> int:
    parser = build_audit_parser()
    args, unknown = parser.parse_known_args(list(argv))

    if not args.all and not args.rule:
        parser.error("Provide a rule name (e.g. 'lelabo audit bp') or use '--all'.")

    env = os.environ.copy()
    env["LELABO_RULE_AUDIT"] = "1"
    env["LELABO_RULE_AUDIT_MODES"] = args.modes
    env["LELABO_RULE_AUDIT_MODEL"] = args.model
    env["LELABO_RULE_AUDIT_EPOCHS"] = str(max(1, int(args.epochs)))
    env["LELABO_RULE_AUDIT_STEPS_PER_EPOCH"] = str(max(1, int(args.steps_per_epoch)))
    if not args.all:
        env["LELABO_RULE_AUDIT_ALGOS"] = str(args.rule).strip()

    cmd = [sys.executable, "-m", "pytest"]
    if not args.no_quiet:
        cmd.append("-q")
    cmd.extend(["-m", "heavy", "tests/test_update_rules_research_audit.py"])
    if args.show_warnings:
        cmd.extend(["-W", "default"])

    extra = list(unknown)
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd.extend(extra)
    return subprocess.call(cmd, env=env)


__all__ = ["build_audit_parser", "run_audit"]
"""Static checks and runtime audits for update rule compatibility."""
