from __future__ import annotations

import importlib
import sys
from argparse import Namespace

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
validator = importlib.import_module("lab.api.train_validator")


def test_supervised_rejects_rl_only_flags() -> None:
    args = Namespace(task="supervised", _provided_flags={"rl_steps", "rl_algo"})
    with pytest.raises(ValueError) as exc:
        validator.validate_train_args(args)
    assert "not valid for task 'supervised'" in str(exc.value)
    assert "--rl-steps" in str(exc.value)


def test_supervised_rejects_rl_param_flag() -> None:
    args = Namespace(task="supervised", _provided_flags={"rl_param"})
    with pytest.raises(ValueError) as exc:
        validator.validate_train_args(args)
    assert "--rl-param" in str(exc.value)


def test_rl_rejects_supervised_only_flags() -> None:
    args = Namespace(task="rl", _provided_flags={"val_frac", "glue_task"})
    with pytest.raises(ValueError) as exc:
        validator.validate_train_args(args)
    assert "not valid for task 'rl'" in str(exc.value)
    assert "--val-frac" in str(exc.value)


def test_rl_model_flag_is_warning_only() -> None:
    args = Namespace(task="rl", _provided_flags={"model"})
    with pytest.warns(UserWarning, match="not implemented yet"):
        validator.validate_train_args(args)
