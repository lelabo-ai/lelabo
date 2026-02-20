from __future__ import annotations

import importlib
import sys
from argparse import Namespace

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
train_api = importlib.import_module("lab.api.train")


def test_task_auto_defaults_to_supervised() -> None:
    args = Namespace(task="auto", source="iris")
    assert train_api._resolve_task(args, dataset_names={"iris", "mnist"}) == "supervised"


def test_task_auto_keeps_legacy_cartpole_rl_behavior() -> None:
    args = Namespace(task="auto", source="CartPole-v1")
    assert train_api._resolve_task(args, dataset_names={"iris", "mnist"}) == "rl"


def test_task_override_is_respected() -> None:
    args = Namespace(task="rl", source="iris")
    assert train_api._resolve_task(args, dataset_names={"iris", "mnist"}) == "rl"


def test_task_auto_errors_when_source_is_ambiguous() -> None:
    args = Namespace(task="auto", source="not_a_dataset_or_env")
    try:
        train_api._resolve_task(args, dataset_names={"iris", "mnist"})
    except ValueError as exc:
        assert "Cannot infer task from source" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ambiguous auto source.")
