from __future__ import annotations

from typing import Any

from ..base import OptimizerUpdateRule
from ...core.steps import compute_loss_and_stats


class Backpropagation(OptimizerUpdateRule):
    """Standard backpropagation with autograd."""

    def train_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        model.train()
        self.zero_grad()

        loss, stats = compute_loss_and_stats(model, task, batch, device)
        loss.backward()
        self.step(model.parameters(), require_grads=True, check_finite_grads=True)

        out = dict(stats)
        out["loss"] = float(loss.item())
        self._mark_step_done()
        return out


# Backward-compatible alias.
Backprop = Backpropagation
