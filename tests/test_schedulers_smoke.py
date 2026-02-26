from __future__ import annotations

import importlib
import math
import sys

import pytest
import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
schedulers_api = importlib.import_module("lelabo.schedulers")


BUILTIN_SCHEDULERS = {
    "steplr",
    "multisteplr",
    "exponentiallr",
    "cosineannealinglr",
    "cosineannealingwarmrestarts",
    "reducelronplateau",
    "onecyclelr",
    "linearlr",
    "constantlr",
}


def _make_optimizer() -> torch.optim.Optimizer:
    model = torch.nn.Linear(4, 2)
    return torch.optim.SGD(model.parameters(), lr=0.1)


def _lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def _changed(before: list[float], after: list[float], *, eps: float = 1e-12) -> bool:
    return any(abs(float(a) - float(b)) > eps for a, b in zip(before, after))


def test_builtin_scheduler_names_include_all_expected() -> None:
    names = set(schedulers_api.get_scheduler_names())
    missing = sorted(BUILTIN_SCHEDULERS - names)
    assert not missing, f"Missing built-in schedulers: {missing}"


@pytest.mark.parametrize(
    ("name", "interval", "epochs", "steps_per_epoch", "monitor", "steps", "kwargs"),
    [
        ("steplr", "epoch", 6, None, "val.loss", 4, {"step_size": 1, "gamma": 0.7}),
        ("multisteplr", "epoch", 6, None, "val.loss", 4, {"milestones": [1, 2, 3], "gamma": 0.7}),
        ("exponentiallr", "epoch", 6, None, "val.loss", 4, {"gamma": 0.8}),
        ("cosineannealinglr", "epoch", 6, None, "val.loss", 4, {"T_max": 4, "eta_min": 0.0}),
        ("cosineannealingwarmrestarts", "epoch", 6, None, "val.loss", 5, {"T_0": 2, "T_mult": 1}),
        ("reducelronplateau", "epoch", 6, None, "val.loss", 4, {"factor": 0.5, "patience": 0, "threshold": 0.0}),
        ("linearlr", "epoch", 6, None, "val.loss", 4, {"start_factor": 1.0, "end_factor": 0.5, "total_iters": 4}),
        ("constantlr", "epoch", 6, None, "val.loss", 5, {"factor": 0.5, "total_iters": 3}),
        ("onecyclelr", "batch", 1, 6, "val.loss", 6, {"max_lr": 0.2}),
    ],
)
def test_builtin_scheduler_smoke_steps_update_lr(
    name: str,
    interval: str,
    epochs: int | None,
    steps_per_epoch: int | None,
    monitor: str,
    steps: int,
    kwargs: dict[str, float | int | list[int]],
) -> None:
    optimizer = _make_optimizer()
    scheduler = schedulers_api.make_scheduler(
        name,
        optimizer,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        interval=interval,
        monitor=monitor,
        **kwargs,
    )
    assert scheduler is not None

    prev = _lrs(optimizer)
    saw_change = False

    for _ in range(int(steps)):
        logs = {"val.loss": 1.0, "train.loss": 1.0}
        stepped = scheduler.step_batch(logs) if interval == "batch" else scheduler.step_epoch(logs)
        assert stepped is True
        cur = _lrs(optimizer)
        assert all(math.isfinite(float(lr)) and float(lr) >= 0.0 for lr in cur)
        if _changed(prev, cur):
            saw_change = True
        prev = cur

    assert saw_change, f"Scheduler '{name}' did not change LR during smoke steps."

