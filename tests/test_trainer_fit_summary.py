from __future__ import annotations

import importlib
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
core_callbacks = importlib.import_module("lelabo.core.callbacks")
trainer_api = importlib.import_module("lelabo.core.trainer")
update_rule_base = importlib.import_module("lelabo.update_rules.base")


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _EpochAwareLearner(update_rule_base.UpdateRule):
    def __init__(self, train_stats_by_epoch: dict[int, dict[str, float]]) -> None:
        super().__init__()
        self.train_stats_by_epoch = {
            int(epoch): {str(k): float(v) for k, v in stats.items()}
            for epoch, stats in train_stats_by_epoch.items()
        }

    def on_train_start(self, model, objective, device, state=None) -> None:
        _ = (model, objective, device, state)

    def train_step(self, model, objective, batch, device, state=None) -> dict[str, float]:
        _ = (objective, batch, device)
        epoch = int(getattr(state, "epoch", 0))
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        self._mark_step_done()
        return dict(self.train_stats_by_epoch[epoch])


def _dummy_loader() -> DataLoader:
    x = torch.zeros(4, 1)
    y = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_trainer_fit_final_metrics_track_last_completed_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    train_stats_by_epoch = {
        1: {"loss": 1.0, "metric": 0.2},
        2: {"loss": 4.0, "metric": 0.6},
        3: {"loss": 2.0, "metric": 0.4},
    }
    val_stats_by_epoch = {
        1: {"loss": 0.9, "acc": 0.1, "metric": 0.1},
        2: {"loss": 0.3, "acc": 0.9, "metric": 0.9},
        3: {"loss": 0.8, "acc": 0.3, "metric": 0.3},
    }

    trainer = trainer_api.Trainer(
        model=_ScalarModel(),
        learner=_EpochAwareLearner(train_stats_by_epoch),
        loss=lambda pred, target: pred.sum() * 0.0,
        device="cpu",
        display_mode="none",
    )
    loader = _dummy_loader()

    def _eval(_loader, *, split=None, emit_report=False, invoke_callbacks=False):
        _ = (_loader, emit_report, invoke_callbacks)
        values = dict(val_stats_by_epoch[int(trainer.state.epoch)])
        metric = float(values.pop("metric"))
        loss = float(values.pop("loss"))
        return trainer_api.SplitSummary(
            split=split,
            loss=loss,
            metric=metric,
            scalars={str(k): float(v) for k, v in values.items()},
            num_samples=4,
            num_batches=2,
            duration_sec=0.01,
        )

    monkeypatch.setattr(trainer, "_evaluate", _eval)
    result = trainer.fit(loader, epochs=3, val_loader=loader)

    assert result.best.source == "default"
    assert result.best.monitor_name == "val.loss"
    assert result.best.monitor_mode == "min"
    assert result.best.best_value == pytest.approx(0.3, rel=1e-6)
    assert result.best.epoch == 2
    assert result.best.train.loss == pytest.approx(4.0, rel=1e-6)
    assert result.best.train.metric == pytest.approx(0.6, rel=1e-6)
    assert result.final_epoch.train.loss == pytest.approx(2.0, rel=1e-6)
    assert result.final_epoch.train.metric == pytest.approx(0.4, rel=1e-6)
    assert result.best.val is not None
    assert result.best.val.loss == pytest.approx(0.3, rel=1e-6)
    assert result.best.val.metric == pytest.approx(0.9, rel=1e-6)
    assert result.final_epoch.val is not None
    assert result.final_epoch.val.loss == pytest.approx(0.8, rel=1e-6)
    assert result.final_epoch.val.metric == pytest.approx(0.3, rel=1e-6)
    assert result.restoration.enabled is False
    assert result.restoration.best_epoch is None
    assert result.restoration.best_checkpoint_available is False
    assert result.restoration.restored_on_train_end is False


def test_trainer_fit_reports_restore_best_without_overwriting_final_epoch_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_stats_by_epoch = {
        1: {"loss": 1.0, "metric": 0.8},
        2: {"loss": 4.0, "metric": 0.2},
    }
    val_stats_by_epoch = {
        1: {"loss": 0.1, "acc": 0.9, "metric": 0.9},
        2: {"loss": 0.9, "acc": 0.1, "metric": 0.1},
    }

    model = _ScalarModel()
    trainer = trainer_api.Trainer(
        model=model,
        learner=_EpochAwareLearner(train_stats_by_epoch),
        loss=lambda pred, target: pred.sum() * 0.0,
        device="cpu",
        display_mode="none",
        callbacks=[
            core_callbacks.EarlyStopping(
                core_callbacks.EarlyStoppingConfig(
                    monitor="val.metric",
                    mode="max",
                    patience=0,
                    warmup_epochs=0,
                    restore_best=True,
                )
            )
        ],
    )
    loader = _dummy_loader()

    def _eval(_loader, *, split=None, emit_report=False, invoke_callbacks=False):
        _ = (_loader, emit_report, invoke_callbacks)
        values = dict(val_stats_by_epoch[int(trainer.state.epoch)])
        metric = float(values.pop("metric"))
        loss = float(values.pop("loss"))
        return trainer_api.SplitSummary(
            split=split,
            loss=loss,
            metric=metric,
            scalars={str(k): float(v) for k, v in values.items()},
            num_samples=4,
            num_batches=2,
            duration_sec=0.01,
        )

    monkeypatch.setattr(trainer, "_evaluate", _eval)
    result = trainer.fit(loader, epochs=5, val_loader=loader)

    assert result.final_epoch.train.loss == pytest.approx(4.0, rel=1e-6)
    assert result.final_epoch.train.metric == pytest.approx(0.2, rel=1e-6)
    assert result.final_epoch.val is not None
    assert result.final_epoch.val.loss == pytest.approx(0.9, rel=1e-6)
    assert result.final_epoch.val.metric == pytest.approx(0.1, rel=1e-6)
    assert result.best.source == "earlystopping"
    assert result.best.monitor_name == "val.metric"
    assert result.best.monitor_mode == "max"
    assert result.best.best_value == pytest.approx(0.9, rel=1e-6)
    assert result.best.epoch == 1
    assert result.restoration.enabled is True
    assert result.restoration.best_epoch == 1
    assert result.restoration.best_checkpoint_available is True
    assert result.restoration.restored_on_train_end is True
    assert float(model.weight.item()) == pytest.approx(1.0, rel=1e-6)
