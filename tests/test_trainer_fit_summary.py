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


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _EpochAwareLearner:
    def __init__(self, train_stats_by_epoch: dict[int, dict[str, float]]) -> None:
        self.train_stats_by_epoch = {
            int(epoch): {str(k): float(v) for k, v in stats.items()}
            for epoch, stats in train_stats_by_epoch.items()
        }

    def on_train_start(self, model, task, device, state=None) -> None:
        _ = (model, task, device, state)

    def train_step(self, model, task, batch, device, state=None) -> dict[str, float]:
        _ = (task, batch, device)
        epoch = int(getattr(state, "epoch", 0))
        with torch.no_grad():
            model.weight.fill_(float(epoch))
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
        task=object(),
        learner=_EpochAwareLearner(train_stats_by_epoch),
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
    result = trainer.fit(loader, epochs=3, show_progress=False, val_loader=loader)

    assert result.best.train_loss == pytest.approx(1.0, rel=1e-6)
    assert result.best.train_metric == pytest.approx(0.6, rel=1e-6)
    assert result.final_epoch.train.loss == pytest.approx(2.0, rel=1e-6)
    assert result.final_epoch.train.metric == pytest.approx(0.4, rel=1e-6)
    assert result.best.val_loss == pytest.approx(0.3, rel=1e-6)
    assert result.best.val_metric == pytest.approx(0.9, rel=1e-6)
    assert result.final_epoch.val is not None
    assert result.final_epoch.val.loss == pytest.approx(0.8, rel=1e-6)
    assert result.final_epoch.val.metric == pytest.approx(0.3, rel=1e-6)
    assert result.best.epoch_by_val == 2
    assert result.restoration.restored_best_model is False
    assert result.restoration.restored_best_epoch is None


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
        task=object(),
        learner=_EpochAwareLearner(train_stats_by_epoch),
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
    result = trainer.fit(loader, epochs=5, show_progress=False, val_loader=loader)

    assert result.final_epoch.train.loss == pytest.approx(4.0, rel=1e-6)
    assert result.final_epoch.train.metric == pytest.approx(0.2, rel=1e-6)
    assert result.final_epoch.val is not None
    assert result.final_epoch.val.loss == pytest.approx(0.9, rel=1e-6)
    assert result.final_epoch.val.metric == pytest.approx(0.1, rel=1e-6)
    assert result.best.epoch_by_val == 1
    assert result.restoration.restored_best_model is True
    assert result.restoration.restored_best_epoch == 1
    assert float(model.weight.item()) == pytest.approx(1.0, rel=1e-6)
