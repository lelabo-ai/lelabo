from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch

from ..seed import seed_everything
from ..core.utils.logger import RunLogger
from ..algorithms.update_rules import get_update_rule_names
from ..metrics import get_metric_names
from ..models import get_model_names
from .train_resolver import _resolve_task, collect_provided_flags, resolve_train_args
from .train_validator import validate_train_args


def _default_device() -> str:
    # Help/arg parsing should not emit CUDA driver warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:.*")
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lelabo train",
        description="Run a LeLabo experiment (supervised or RL).",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Training source. Dataset name for supervised or Gym env id for RL.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "supervised", "rl"],
        help="Training task selection: auto, supervised, or rl.",
    )
    parser.add_argument("--model", choices=get_model_names(), default="cnn")
    parser.add_argument("--algo", choices=get_update_rule_names(), default="kp2")
    available_metrics = get_metric_names()
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help=(
            "Comma-separated optional training metrics. "
            f"Available: {', '.join(available_metrics) if available_metrics else '(none)'}"
        ),
    )
    parser.add_argument(
        "--bp-alignment-every",
        type=int,
        default=1,
        help="Compute BP-alignment diagnostics every N local batches when requested metrics need it.",
    )
    parser.add_argument(
        "--bp-alignment-eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon for BP-alignment diagnostics.",
    )
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--determinism",
        type=str,
        default="relaxed",
        choices=["off", "relaxed", "strict"],
        help="Determinism policy: off, relaxed, strict.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "sgd+momentum", "ano"])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        help="torch scheduler name: none|step|multistep|cosine|plateau|onecycle|exponential|constant|linear",
    )
    parser.add_argument(
        "--lr-scheduler-interval",
        type=str,
        default="epoch",
        choices=["epoch", "batch"],
        help="when to call scheduler.step()",
    )
    parser.add_argument(
        "--lr-scheduler-monitor",
        type=str,
        default="val.loss",
        help="metric key used by ReduceLROnPlateau (e.g. val.loss or val.metric)",
    )
    parser.add_argument(
        "--lr-scheduler-kwargs",
        type=str,
        default=None,
        help="JSON dict of scheduler kwargs, e.g. '{\"step_size\":10,\"gamma\":0.5}'",
    )
    parser.add_argument("--input-noise-training", type=float, default=0.0, help="stddev gaussian noise on inputs during training")
    parser.add_argument("--input-noise-dataset", type=float, default=0.0, help="stddev gaussian noise on inputs in dataset (train+val+test)")
    parser.add_argument("--noise-on-test", type=int, choices=[0, 1], default=0, help="0/1 to add noise to test set if input-noise-dataset > 0")
    parser.add_argument("--val-frac", type=float, default=0.1, help="fraction of train set used as validation (0 disables)")
    parser.add_argument("--early-stop", action="store_true", help="enable early stopping")
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false", help="disable early stopping")
    parser.set_defaults(early_stop=True)
    parser.add_argument("--early-monitor", type=str, default="val.acc")
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--early-min-delta", type=float, default=0.0)
    parser.add_argument("--early-warmup", type=int, default=5)
    parser.add_argument("--rl-algo", type=str, default="ppo", choices=["dqn", "ppo"])
    parser.add_argument("--rl-steps", type=int, default=500_000)
    parser.add_argument("--rl-eval-episodes", type=int, default=10)
    parser.add_argument(
        "--rl-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "RL algo config override. Repeatable. "
            "Examples: --rl-param gamma=0.97 --rl-param num_steps=256"
        ),
    )
    parser.add_argument("--robustness", type=str, default="input_noise", choices=["none", "input_noise", "relative_input_noise", "weight_noise", "all"])
    parser.add_argument("--noise-trials", type=int, default=30)
    parser.add_argument(
        "--glue-task",
        type=str,
        default="sst2",
        choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"],
    )
    parser.add_argument("--hf-model", type=str, default="bert-base-uncased")
    parser.add_argument("--hf-trust-remote-code", action="store_true", help="allow HF trust_remote_code for custom models")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--run-dir", type=str, default=None, help="writes metrics.jsonl + meta.json + summary.json")
    return parser


def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = build_train_parser().parse_args(raw_argv)
    args._provided_flags = collect_provided_flags(raw_argv)
    return args


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    seed_state = seed_everything(args.seed, mode=args.determinism)

    run_dir = Path(args.run_dir) if args.run_dir else None
    logger = RunLogger(run_dir=run_dir)
    logger.write_meta(vars(args))
    logger.log(
        {
            "t": "seed",
            "seed": int(seed_state.seed),
            "determinism": str(seed_state.mode),
            "deterministic_algorithms": bool(seed_state.deterministic_algorithms),
        }
    )

    task = str(getattr(args, "task", "supervised")).strip().lower()
    if task == "rl":
        from .train_rl import run_rl

        rl_summary = run_rl(args, logger)
        summary = {"args": vars(args), "rl": {"algo": args.rl_algo, **rl_summary}}
        logger.write_summary(summary)
        return summary

    from ..core.runners.supervised_runner import run_supervised

    summary = run_supervised(args, logger)
    logger.write_summary(summary)
    return summary


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    resolved = resolve_train_args(args)
    validate_train_args(resolved)
    return run_experiment(resolved)


def run_train_from_argv(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run_train(parse_train_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_train_from_argv(argv)
    except ValueError as exc:
        raise SystemExit(str(exc))
    return 0
