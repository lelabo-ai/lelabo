from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch

from ..config.resolve import (
    resolve_rl_config,
    resolve_supervised_config,
    to_rl_namespace,
    to_supervised_namespace,
)
from ..core.utils.seed import seed_everything
from ..core.utils.logger import RunLogger
from ..supervised.datasets import get_dataset_names
from ..update_rules import get_update_rule_names
from ..models import get_model_names
from ..optimizers import get_optimizer_names
from ..rl.algorithms import get_rl_algo_names
from .train_rl_config import parse_rl_param_overrides


def _default_device() -> str:
    # Help/arg parsing should not emit CUDA driver warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:.*")
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    cur = target
    for key in path[:-1]:
        child = cur.get(key)
        if not isinstance(child, dict):
            child = {}
            cur[key] = child
        cur = child
    cur[path[-1]] = value


def _resolve_mode_config_path(mode: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    cwd = Path.cwd()
    candidates = [
        cwd / f"train.{mode}.toml",
        cwd / "train.toml",
        cwd / "configs" / f"train.{mode}.toml",
        cwd / "configs" / "train.toml",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)
    return None


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional train TOML config file. If omitted, LeLabo auto-detects: "
            "train.<mode>.toml, train.toml, configs/train.<mode>.toml, configs/train.toml."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Advanced nested override (repeatable), e.g. "
            "--set model.params.hidden=1024 --set scheduler.params.gamma=0.5"
        ),
    )


def _add_runtime_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--determinism",
        type=str,
        default=None,
        choices=["off", "relaxed", "strict"],
    )
    parser.add_argument("--verbose", type=int, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Writes metrics.jsonl + meta.json + summary.json.",
    )


def _add_supervised_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=list(get_dataset_names()),
        type=str,
        default=None,
    )
    parser.add_argument("--model", choices=list(get_model_names()), default=None)
    parser.add_argument("--algo", choices=list(get_update_rule_names()), default=None)
    parser.add_argument("--optimizer", choices=list(get_optimizer_names()), default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--metrics", type=str, default=None, help="Comma-separated metrics list.")


def _add_rl_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        metavar="ENV_ID",
    )
    parser.add_argument("--algo", choices=list(get_update_rule_names()), default=None)
    parser.add_argument("--optimizer", choices=list(get_optimizer_names()), default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--rl-algo", type=str, default=None, choices=list(get_rl_algo_names()))
    parser.add_argument("--rl-steps", type=int, default=None)
    parser.add_argument("--rl-eval-episodes", type=int, default=None)
    parser.add_argument(
        "--rl-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lelabo train",
        description="Run a LeLabo experiment.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    supervised_parser = subparsers.add_parser(
        "supervised",
        help="Run supervised training.",
        description="Run a supervised LeLabo experiment from a layered config.",
    )
    _add_config_args(supervised_parser)
    _add_supervised_overrides(supervised_parser)
    _add_runtime_override_args(supervised_parser)

    rl_parser = subparsers.add_parser(
        "rl",
        help="Run reinforcement-learning training.",
        description="Run an RL LeLabo experiment from a layered config.",
    )
    _add_config_args(rl_parser)
    _add_rl_overrides(rl_parser)
    _add_runtime_override_args(rl_parser)

    return parser


def _build_supervised_cli_overrides(parsed: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if parsed.dataset is not None:
        _set_nested(out, ["dataset", "name"], parsed.dataset)
    if parsed.model is not None:
        _set_nested(out, ["model", "name"], parsed.model)
    if parsed.algo is not None:
        _set_nested(out, ["update_rule", "name"], parsed.algo)
    if parsed.optimizer is not None:
        _set_nested(out, ["optimizer", "name"], parsed.optimizer)
    if parsed.lr is not None:
        _set_nested(out, ["optimizer", "params", "lr"], parsed.lr)
    if parsed.weight_decay is not None:
        _set_nested(out, ["optimizer", "params", "weight_decay"], parsed.weight_decay)
    if parsed.epochs is not None:
        _set_nested(out, ["train", "epochs"], parsed.epochs)
    if parsed.batch is not None:
        _set_nested(out, ["train", "batch"], parsed.batch)

    if parsed.metrics is not None:
        metrics = [tok.strip() for tok in str(parsed.metrics).split(",") if tok.strip()]
        _set_nested(out, ["metrics"], [{"name": name} for name in metrics])

    if parsed.device is not None:
        _set_nested(out, ["runtime", "device"], parsed.device)
    if parsed.seed is not None:
        _set_nested(out, ["runtime", "seed"], parsed.seed)
    if parsed.determinism is not None:
        _set_nested(out, ["runtime", "determinism"], parsed.determinism)
    if parsed.verbose is not None:
        _set_nested(out, ["runtime", "verbose"], parsed.verbose)
    if parsed.run_dir is not None:
        _set_nested(out, ["runtime", "run_dir"], parsed.run_dir)

    return out


def _build_rl_cli_overrides(parsed: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if parsed.env is not None:
        _set_nested(out, ["env"], parsed.env)
    if parsed.algo is not None:
        _set_nested(out, ["update_rule", "name"], parsed.algo)
    if parsed.optimizer is not None:
        _set_nested(out, ["optimizer", "name"], parsed.optimizer)
    if parsed.lr is not None:
        _set_nested(out, ["optimizer", "params", "lr"], parsed.lr)
    if parsed.weight_decay is not None:
        _set_nested(out, ["optimizer", "params", "weight_decay"], parsed.weight_decay)
    if parsed.rl_algo is not None:
        _set_nested(out, ["rl", "algo"], parsed.rl_algo)
    if parsed.rl_steps is not None:
        _set_nested(out, ["rl", "steps"], parsed.rl_steps)
    if parsed.rl_eval_episodes is not None:
        _set_nested(out, ["rl", "eval_episodes"], parsed.rl_eval_episodes)
    if parsed.rl_param:
        _set_nested(out, ["rl", "params"], parse_rl_param_overrides(parsed.rl_param))

    if parsed.device is not None:
        _set_nested(out, ["runtime", "device"], parsed.device)
    if parsed.seed is not None:
        _set_nested(out, ["runtime", "seed"], parsed.seed)
    if parsed.determinism is not None:
        _set_nested(out, ["runtime", "determinism"], parsed.determinism)
    if parsed.verbose is not None:
        _set_nested(out, ["runtime", "verbose"], parsed.verbose)
    if parsed.run_dir is not None:
        _set_nested(out, ["runtime", "run_dir"], parsed.run_dir)
    return out


def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = build_train_parser().parse_args(raw_argv)
    if parsed.mode == "supervised":
        config_path = _resolve_mode_config_path("supervised", parsed.config)
        cfg = resolve_supervised_config(
            config_path=config_path,
            cli_overrides=_build_supervised_cli_overrides(parsed),
            set_overrides=list(parsed.set or []),
        )
        return to_supervised_namespace(cfg, device_resolver=_default_device)
    if parsed.mode == "rl":
        config_path = _resolve_mode_config_path("rl", parsed.config)
        cfg = resolve_rl_config(
            config_path=config_path,
            cli_overrides=_build_rl_cli_overrides(parsed),
            set_overrides=list(parsed.set or []),
        )
        return to_rl_namespace(cfg, device_resolver=_default_device)
    raise ValueError(f"Unknown train mode '{parsed.mode}'.")


def _normalize_train_args(args: argparse.Namespace) -> argparse.Namespace:
    mode = str(getattr(args, "mode", "")).strip().lower()
    if mode == "supervised":
        dataset = str(getattr(args, "dataset", "")).strip()
        if not dataset:
            raise ValueError("--dataset cannot be empty.")
        args.task = "supervised"
        return args

    if mode == "rl":
        env = str(getattr(args, "env", "")).strip()
        if not env:
            raise ValueError("--env cannot be empty.")
        args.task = "rl"
        args.dataset = f"env:{env}"
        return args

    raise ValueError(f"Unknown train mode '{mode}'. Expected one of: supervised, rl.")


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

    task = str(getattr(args, "task", "")).strip().lower()
    if task == "rl":
        from .train_rl import run_rl_train

        rl_summary = run_rl_train(args, logger)
        summary = {"args": vars(args), "rl": {"algo": args.rl_algo, **rl_summary}}
        logger.write_summary(summary)
        return summary

    if task == "supervised":
        from .train_supervised import run_supervised_train

        summary = run_supervised_train(args, logger)
        logger.write_summary(summary)
        return summary

    raise ValueError(f"Unknown task '{task}'.")


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    normalized = _normalize_train_args(args)
    return run_experiment(normalized)


def run_train_from_argv(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run_train(parse_train_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_train_from_argv(argv)
    except ValueError as exc:
        raise SystemExit(str(exc))
    return 0
