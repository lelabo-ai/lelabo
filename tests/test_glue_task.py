from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
task_mod = importlib.import_module("lelabo.core.task")


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


def test_glue_task_evaluate_classification() -> None:
    task = task_mod.GLUETask(task_name="sst2", is_regression=False, num_labels=2)
    model = _DummyGlueClassifier(num_labels=2)

    loader = [
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "labels": torch.tensor([0, 1], dtype=torch.long),
        }
    ]

    out = task.evaluate(model, loader, device="cpu")
    assert "loss" in out
    assert "acc" in out
    assert out["acc"] >= 0.99


def test_glue_task_evaluate_regression() -> None:
    task = task_mod.GLUETask(task_name="stsb", is_regression=True, num_labels=1)
    model = _DummyGlueRegressor()

    loader = [
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "labels": torch.tensor([1.5, 2.0], dtype=torch.float32),
        }
    ]

    out = task.evaluate(model, loader, device="cpu")
    assert "loss" in out
    assert "mse" in out
    assert "metric" in out
