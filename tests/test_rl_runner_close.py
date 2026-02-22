from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
runner_mod = importlib.import_module("lab.core.runners.rl_runner")


class _DummyEnv:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _DummyAlgo:
    def __init__(self) -> None:
        self.total_steps = 0

    def to(self, _device: str) -> None:
        return None

    def collect(self, _env, *, device: str):
        self.total_steps += 1
        return {}, {"episodes": []}

    def update(self, _batch, *, device: str):
        return {"loss": 1.0}

    def evaluate(self, _env, *, eval_episodes: int, device: str):
        return {"mean_return": 0.0, "n": int(eval_episodes)}


class _FailingAlgo(_DummyAlgo):
    def update(self, _batch, *, device: str):
        raise RuntimeError("boom")


def test_rl_runner_closes_train_and_eval_envs_on_success() -> None:
    train_env = _DummyEnv()
    eval_env = _DummyEnv()
    runner = runner_mod.RLRunner(train_env=train_env, algo=_DummyAlgo(), device="cpu", verbose=False)

    out = runner.train(total_steps=3, eval_env=eval_env, eval_episodes=2)

    assert out["total_steps"] == 3
    assert train_env.closed == 1
    assert eval_env.closed == 1


def test_rl_runner_closes_envs_on_failure() -> None:
    train_env = _DummyEnv()
    eval_env = _DummyEnv()
    runner = runner_mod.RLRunner(train_env=train_env, algo=_FailingAlgo(), device="cpu", verbose=False)

    with pytest.raises(RuntimeError, match="boom"):
        runner.train(total_steps=3, eval_env=eval_env, eval_episodes=2)

    assert train_env.closed == 1
    assert eval_env.closed == 1
