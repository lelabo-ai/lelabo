from __future__ import annotations

import importlib
import sys

import pytest
import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
schedulers_api = importlib.import_module("lab.schedulers")


def _make_optimizer() -> torch.optim.Optimizer:
    model = torch.nn.Linear(4, 2)
    return torch.optim.SGD(model.parameters(), lr=0.1)


def test_builtin_step_scheduler_wraps_and_steps_by_epoch() -> None:
    optimizer = _make_optimizer()
    scheduler = schedulers_api.make_scheduler(
        "steplr",
        optimizer,
        epochs=3,
        interval="epoch",
        step_size=1,
        gamma=0.5,
    )
    assert scheduler is not None

    lr0 = float(optimizer.param_groups[0]["lr"])
    scheduler.step_batch({"loss": 1.0})
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(lr0, rel=1e-8)

    scheduler.step_epoch({"train.loss": 1.0})
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(lr0 * 0.5, rel=1e-8)


def test_onecycle_requires_batch_interval() -> None:
    optimizer = _make_optimizer()
    with pytest.raises(ValueError):
        schedulers_api.make_scheduler(
            "onecycle",
            optimizer,
            epochs=2,
            steps_per_epoch=4,
            interval="epoch",
            max_lr=0.2,
        )


def test_scheduler_can_be_loaded_from_capsule_plugin(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_with_scheduler"
    (capsule_root / "schedulers").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_with_scheduler\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "schedulers" / "example.py").write_text(
        "import torch\n"
        "from lab.schedulers import SchedulerContext, register_scheduler\n\n"
        "@register_scheduler('capsule_step_lr')\n"
        "def build_capsule_step_lr(ctx: SchedulerContext):\n"
        "    return torch.optim.lr_scheduler.StepLR(ctx.optimizer, step_size=1, gamma=0.9)\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(capsule_root)
    optimizer = _make_optimizer()
    scheduler = schedulers_api.make_scheduler("capsule_step_lr", optimizer, interval="epoch")
    assert scheduler is not None
    scheduler.step_epoch({"train.loss": 1.0})
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(0.09, rel=1e-8)
