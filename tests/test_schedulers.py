from __future__ import annotations

import importlib
import sys

import pytest
import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
schedulers_api = importlib.import_module("lelabo.schedulers")


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


def test_cosineannealing_requires_resolvable_horizon_without_t_max() -> None:
    optimizer = _make_optimizer()
    with pytest.raises(ValueError, match="CosineAnnealingLR requires 'T_max'"):
        schedulers_api.make_scheduler(
            "cosineannealinglr",
            optimizer,
            interval="epoch",
        )


def test_linearlr_requires_resolvable_horizon_without_total_iters() -> None:
    optimizer = _make_optimizer()
    with pytest.raises(ValueError, match="LinearLR requires 'total_iters'"):
        schedulers_api.make_scheduler(
            "linearlr",
            optimizer,
            interval="batch",
            epochs=4,
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
        "from lelabo.schedulers import SchedulerContext, register_scheduler\n\n"
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


def test_multistep_ratio_offset_matches_legacy_softhebb_schedule() -> None:
    optimizer = _make_optimizer()
    scheduler = schedulers_api.make_scheduler(
        "multisteplr",
        optimizer,
        epochs=51,
        interval="epoch",
        gamma=0.5,
        milestone_ratios=[0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9],
        milestone_offset=1,
    )
    assert scheduler is not None

    changed_at: list[int] = []
    prev_lr = float(optimizer.param_groups[0]["lr"])
    for ep in range(1, 52):
        scheduler.step_epoch({"train.loss": 1.0})
        cur_lr = float(optimizer.param_groups[0]["lr"])
        if abs(cur_lr - prev_lr) > 1e-12:
            changed_at.append(ep)
            prev_lr = cur_lr

    assert changed_at == [11, 18, 26, 31, 36, 41, 46]


def test_multistep_rejects_mixed_milestones_and_ratios() -> None:
    optimizer = _make_optimizer()
    with pytest.raises(ValueError):
        schedulers_api.make_scheduler(
            "multisteplr",
            optimizer,
            epochs=10,
            interval="epoch",
            milestones=[3, 6],
            milestone_ratios=[0.5],
        )


def test_multistep_ratios_require_epochs_or_explicit_horizon() -> None:
    optimizer = _make_optimizer()
    with pytest.raises(ValueError):
        schedulers_api.make_scheduler(
            "multisteplr",
            optimizer,
            interval="epoch",
            milestone_ratios=[0.5],
        )
