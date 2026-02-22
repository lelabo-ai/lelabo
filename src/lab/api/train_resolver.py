from __future__ import annotations

import argparse
import re
from typing import Sequence

from ..supervised.datasets import get_dataset_names


def collect_provided_flags(argv: Sequence[str]) -> set[str]:
    provided: set[str] = set()
    for token in list(argv):
        raw = str(token)
        if raw == "--":
            break
        if not raw.startswith("--"):
            continue
        name = raw[2:].split("=", 1)[0].strip()
        if not name:
            continue
        provided.add(name.replace("-", "_"))
    return provided


def _looks_like_env_id(source: str) -> bool:
    return bool(re.search(r"-v[0-9]+$", source.strip()))


def _resolve_task(args: argparse.Namespace, dataset_names: set[str] | None = None) -> str:
    requested = str(getattr(args, "task", "auto")).strip().lower()
    if requested in {"supervised", "rl"}:
        return requested

    names = dataset_names or set(get_dataset_names())
    source = str(getattr(args, "source", "")).strip()
    if source.lower() in {name.lower() for name in names}:
        return "supervised"
    if _looks_like_env_id(source):
        return "rl"

    raise ValueError(
        f"Cannot infer task from source '{source}'. "
        "Use a known dataset name, a Gym env id like 'CartPole-v1', "
        "or force '--task supervised|rl'."
    )


def resolve_train_args(args: argparse.Namespace) -> argparse.Namespace:
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
