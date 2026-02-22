from __future__ import annotations

import importlib
import sys
from argparse import Namespace

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
train_api = importlib.import_module("lab.api.train")
update_rules_api = importlib.import_module("lab.update_rules")


def test_supervised_mode_normalization_sets_task() -> None:
    args = Namespace(mode="supervised", dataset="iris")
    out = train_api._normalize_train_args(args)
    assert out.task == "supervised"
    assert out.dataset == "iris"


def test_rl_mode_normalization_sets_task_and_dataset() -> None:
    args = Namespace(mode="rl", env="CartPole-v1")
    out = train_api._normalize_train_args(args)
    assert out.task == "rl"
    assert out.env == "CartPole-v1"
    assert out.dataset == "env:CartPole-v1"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown train mode"):
        train_api._normalize_train_args(Namespace(mode="auto", source="iris"))


def test_legacy_unified_source_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown train mode"):
        train_api._normalize_train_args(Namespace(source="iris"))


def test_supervised_default_algo_is_registered() -> None:
    args = train_api.parse_train_args(["supervised", "--dataset", "iris"])
    available = set(update_rules_api.get_update_rule_names())
    assert args.algo in available


def test_supervised_robustness_max_samples_arg_parses() -> None:
    args = train_api.parse_train_args(
        ["supervised", "--dataset", "iris", "--robustness-max-samples", "123"]
    )
    assert args.robustness_max_samples == 123
