from __future__ import annotations

import importlib
import sys

import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
task_api = importlib.import_module("lelabo.core.task")
loss_api = importlib.import_module("lelabo.losses")


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(labels.size(0), num_classes, dtype=torch.float32)
    out.scatter_(1, labels.view(-1, 1), 1.0)
    return out


def test_classification_task_uses_configured_bce_with_logits_loss() -> None:
    logits = torch.tensor(
        [
            [0.5, -1.0, 2.0],
            [1.2, 0.1, -0.4],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([2, 0], dtype=torch.long)
    targets = _one_hot(labels, num_classes=3).to(dtype=logits.dtype)

    loss_fn = loss_api.make_loss("bce_with_logits")
    task = task_api.ClassificationTask(
        num_classes=3,
        loss_name="bce_with_logits",
        loss_fn=loss_fn,
    )

    got_loss = task.loss(logits, labels)
    exp_loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.allclose(got_loss, exp_loss)

    got_delta = task.output_deltas(logits, labels)["logits"]
    exp_delta = (torch.sigmoid(logits) - targets) / float(logits.numel())
    assert torch.allclose(got_delta, exp_delta)

