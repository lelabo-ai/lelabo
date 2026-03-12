from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from types import SimpleNamespace

import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
loss_api = importlib.import_module("lelabo.losses")
metrics_api = importlib.import_module("lelabo.metrics")
trainer_api = importlib.import_module("lelabo.core.trainer")
update_rule_base = importlib.import_module("lelabo.update_rules.base")


class _EvalOnlyLearner(update_rule_base.UpdateRule):
    def train_step(self, model, objective, batch, device, state=None):
        _ = (model, objective, batch, device, state)
        raise RuntimeError("This test only exercises Trainer.evaluate().")


class _DummyGlueClassifier(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = int(num_labels)

    def forward(self, **batch):
        labels = batch["labels"].long()
        logits = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float() * 5.0
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


class _DummyGlueRegressor(torch.nn.Module):
    def forward(self, **batch):
        labels = batch["labels"].float().view(-1, 1)
        logits = labels + 0.25
        loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1))
        return SimpleNamespace(logits=logits, loss=loss)


def test_trainer_evaluate_glue_classification_batch() -> None:
    model = _DummyGlueClassifier(num_labels=2)
    trainer = trainer_api.Trainer(
        model=model,
        learner=_EvalOnlyLearner(),
        loss=loss_api.make_loss("ce"),
        device="cpu",
        display_mode="none",
    )

    loader = [
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }
    ]

    out = trainer.evaluate(loader, split="val")
    assert out.loss >= 0.0
    assert out.metric is not None
    assert out.metric >= 0.99
    assert out.scalars["acc"] >= 0.99


def test_trainer_evaluate_glue_regression_batch() -> None:
    model = _DummyGlueRegressor()
    metric_ctx = metrics_api.MetricContext(
        args=Namespace(),
        mode="supervised",
        dataset="glue",
        algo="bp",
        extra={"metric_params": {}},
    )
    trainer = trainer_api.Trainer(
        model=model,
        learner=_EvalOnlyLearner(),
        loss=loss_api.make_loss("mse"),
        device="cpu",
        display_mode="none",
        metrics=[
            metrics_api.build_metric("mse", metric_ctx),
            metrics_api.build_metric("mae", metric_ctx),
        ],
    )

    loader = [
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "labels": torch.tensor([1.5, 2.0], dtype=torch.float32),
        }
    ]

    out = trainer.evaluate(loader, split="val")
    assert out.loss >= 0.0
    assert "mse" in out.scalars
    assert "mae" in out.scalars
