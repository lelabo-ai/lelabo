from __future__ import annotations

import importlib
import sys
from argparse import Namespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
metrics_api = importlib.import_module("lelabo.metrics")
metrics_payload = importlib.import_module("lelabo.metrics.payload")
reporters_api = importlib.import_module("lelabo.core.reporters")
trainer_api = importlib.import_module("lelabo.core.trainer")
loss_api = importlib.import_module("lelabo.losses")
backprop_api = importlib.import_module("lelabo.update_rules.builtins.backprop")


def test_builtin_metric_names_are_registered() -> None:
    names = set(metrics_api.get_metric_names())
    for required in {"acc", "precision", "recall", "f1", "mse", "mae", "rmse", "r2"}:
        assert required in names


def test_builtin_f1_macro_metric_computes_from_payload() -> None:
    ctx = metrics_api.MetricContext(
        args=Namespace(),
        mode="supervised",
        dataset="iris",
        algo="bp",
        extra={"metric_params": {"f1": {"average": "macro"}}},
    )
    metric = metrics_api.build_metric("f1", ctx)
    metric.reset("train", None)
    metric.update(
        "train",
        stats={
            metrics_payload.METRIC_Y_TRUE_KEY: torch.tensor([0, 1, 1, 0]),
            metrics_payload.METRIC_Y_PRED_KEY: torch.tensor([0, 1, 0, 0]),
            metrics_payload.METRIC_KIND_KEY: "classification",
        },
        batch_size=4,
        state=None,
    )
    out = metric.compute("train", None)
    assert "f1_macro" in out
    assert out["f1_macro"] == pytest.approx((0.8 + (2.0 / 3.0)) / 2.0, rel=1e-6)


def test_trainer_reports_builtin_eval_metrics() -> None:
    torch.manual_seed(0)
    x = torch.randn(24, 4)
    y = (x[:, 0] > 0).long()
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=8, shuffle=False)
    val_loader = DataLoader(ds, batch_size=8, shuffle=False)

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    learner = backprop_api.Backprop(optimizer=optimizer)
    loss_fn = loss_api.make_loss("ce")

    ctx = metrics_api.MetricContext(
        args=Namespace(),
        mode="supervised",
        dataset="iris",
        algo="bp",
        extra={"metric_params": {"f1": {"average": "macro"}}},
    )
    metric = metrics_api.build_metric("f1", ctx)

    trainer = trainer_api.Trainer(
        model=model,
        learner=learner,
        loss=loss_fn,
        device="cpu",
        display_mode="none",
        metrics=[metric],
    )

    train_out = trainer.fit(train_loader, epochs=1, val_loader=val_loader)
    assert "f1_macro" in train_out.final_epoch.train.scalars

    eval_out = trainer.evaluate(val_loader, split="val")
    assert "f1_macro" in eval_out.scalars


def test_trainer_warns_when_rich_requested_but_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = (x[:, 0] > 0).long()
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    learner = backprop_api.Backprop(optimizer=optimizer)
    loss_fn = loss_api.make_loss("ce")

    monkeypatch.setattr(reporters_api, "Console", None, raising=True)
    monkeypatch.setattr(reporters_api, "Table", None, raising=True)

    with pytest.warns(UserWarning, match="display='rich'.*pip install rich"):
        trainer = trainer_api.Trainer(
            model=model,
            learner=learner,
            loss=loss_fn,
            device="cpu",
            display_mode="rich",
            metrics=[],
        )
    assert trainer.display_mode == "compact"


def test_trainer_rejects_invalid_display_mode() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    learner = backprop_api.Backprop(optimizer=optimizer)
    loss_fn = loss_api.make_loss("ce")

    with pytest.raises(ValueError, match="display_mode must be one of: none, compact, rich"):
        trainer_api.Trainer(
            model=model,
            learner=learner,
            loss=loss_fn,
            device="cpu",
            display_mode="quiet",
            metrics=[],
        )


def test_metric_build_rejects_invalid_positive_label() -> None:
    ctx = metrics_api.MetricContext(
        args=Namespace(),
        mode="supervised",
        dataset="iris",
        algo="bp",
        extra={"metric_params": {"precision": {"average": "binary", "positive_label": "oops"}}},
    )
    with pytest.raises(ValueError, match="positive_label"):
        metrics_api.build_metric("precision", ctx)


def test_metric_build_rejects_invalid_average() -> None:
    ctx = metrics_api.MetricContext(
        args=Namespace(),
        mode="supervised",
        dataset="iris",
        algo="bp",
        extra={"metric_params": {"f1": {"average": "weigthted"}}},
    )
    with pytest.raises(ValueError, match="Invalid metric param 'average'"):
        metrics_api.build_metric("f1", ctx)
