from __future__ import annotations

import importlib
import sys

import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
loss_api = importlib.import_module("lelabo.losses")
signal_api = importlib.import_module("lelabo.update_rules.teaching_signals")


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(labels.size(0), num_classes, dtype=torch.float32)
    out.scatter_(1, labels.view(-1, 1), 1.0)
    return out


def test_bce_teaching_signal_matches_analytic_delta() -> None:
    logits = torch.tensor(
        [
            [0.5, -1.0, 2.0],
            [1.2, 0.1, -0.4],
        ],
        dtype=torch.float32,
    )
    probs = torch.sigmoid(logits)
    labels = torch.tensor([2, 0], dtype=torch.long)
    targets = _one_hot(labels, num_classes=3).to(dtype=probs.dtype)

    loss_fn = loss_api.make_loss("bce")
    got_loss = loss_fn(probs, labels)
    exp_loss = torch.nn.BCELoss()(probs, targets)
    assert torch.allclose(got_loss, exp_loss)

    got_delta = signal_api.logits_delta_from_loss(loss_fn, probs, labels, rule_name="FA")
    exp_delta = (probs - targets) / (probs * (1.0 - probs))
    exp_delta = exp_delta / float(probs.numel())
    assert torch.allclose(got_delta, exp_delta)
