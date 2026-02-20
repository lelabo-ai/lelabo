from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch

from ..seed import seed_everything
from ..core.utils.logger import RunLogger
from ..algorithms.update_rules import get_update_rule_names
from ..metrics import get_metric_names
from ..models import get_model_names
from ..datasets import get_dataset_names


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
    parser.add_argument("--rl-steps", type=int, default=500_000)
    parser.add_argument("--rl-eval-episodes", type=int, default=10)
    parser.add_argument("--rl-algo", type=str, default="ppo", choices=["dqn", "ppo"])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=50_000)
    parser.add_argument("--rl-batch-size", type=int, default=256)
    parser.add_argument("--ppo-num-envs", type=int, default=8)
    parser.add_argument("--ppo-num-steps", type=int, default=128)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-num-minibatches", type=int, default=4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--clip-vloss", type=int, default=1)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
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
    args = build_train_parser().parse_args(list(argv) if argv is not None else None)
    return _resolve_source(args)


def _looks_like_env_id(source: str) -> bool:
    return bool(re.search(r"-v[0-9]+$", source.strip()))


def _resolve_task(args: argparse.Namespace, dataset_names: set[str]) -> str:
    requested = str(getattr(args, "task", "auto")).strip().lower()
    if requested in {"supervised", "rl"}:
        return requested

    source = str(getattr(args, "source", "")).strip()
    if source.lower() in {name.lower() for name in dataset_names}:
        return "supervised"
    if _looks_like_env_id(source):
        return "rl"

    raise ValueError(
        f"Cannot infer task from source '{source}'. "
        "Use a known dataset name, a Gym env id like 'CartPole-v1', "
        "or force '--task supervised|rl'."
    )


def _resolve_source(args: argparse.Namespace) -> argparse.Namespace:
    source = str(getattr(args, "source", "")).strip()
    if not source:
        raise ValueError("--source cannot be empty.")

    dataset_names = set(get_dataset_names())
    dataset_by_lower = {name.lower(): name for name in dataset_names}
    task = _resolve_task(args, dataset_names=dataset_names)
    args.task = task

    if task == "supervised":
        dataset = dataset_by_lower.get(source.lower())
        if dataset is None:
            raise ValueError(
                f"Unknown supervised dataset source '{source}'. "
                f"Available datasets: {sorted(dataset_names)}"
            )
        args.dataset = dataset
        args.env = None
        return args

    args.env = source
    args.dataset = f"env:{source}"
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
    return run_experiment(_resolve_source(args))


def run_train_from_argv(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run_train(parse_train_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    run_train_from_argv(argv)
    return 0
