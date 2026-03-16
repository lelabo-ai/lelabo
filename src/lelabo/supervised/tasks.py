"""Task metadata and defaults for supervised learning problems."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from ..metrics.payload import build_metric_payload


def _onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(labels.size(0), num_classes, device=labels.device, dtype=torch.float32)
    out.scatter_(1, labels.view(-1, 1), 1.0)
    return out


def _as_class_indices(y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 2 and int(y.size(1)) == 1 and not torch.is_floating_point(y):
        return y.view(-1).long()
    if y.dim() == 2 and torch.is_floating_point(y):
        return y.argmax(dim=1).long()
    if y.dim() == 1:
        return y.long()
    raise ValueError(f"Expected labels shaped [B], [B,1], or one-hot [B,C], got {tuple(y.shape)}.")


class ClassificationTask:
    def __init__(
        self,
        num_classes: int,
        *,
        loss_name: str = "cross_entropy",
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        self.num_classes = int(num_classes)
        self.loss_name = str(loss_name).strip().lower()
        if loss_fn is None:
            self.loss_fn = lambda logits, y: F.cross_entropy(logits, _as_class_indices(y))
        else:
            self.loss_fn = loss_fn

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, y)

    @torch.no_grad()
    def metrics(self, logits: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
        y_idx = _as_class_indices(y)
        preds = logits.argmax(dim=1)
        acc = (preds == y_idx).float().mean().item()
        out: dict[str, Any] = {"acc": float(acc)}
        out.update(build_metric_payload(y_true=y_idx, y_pred=preds, kind="classification"))
        return out

class GLUETask:
    def __init__(self, task_name: str, is_regression: bool, num_labels: int):
        self.task_name = str(task_name).lower()
        self.is_regression = bool(is_regression)
        self.num_labels = int(num_labels)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.is_regression:
            preds = logits.view(-1).float()
            targets = y.view(-1).float()
            return F.mse_loss(preds, targets)
        return F.cross_entropy(logits, y.long())

    @torch.no_grad()
    def metrics(self, outputs_or_logits: Any, y: torch.Tensor) -> dict[str, Any]:
        logits = outputs_or_logits.logits if hasattr(outputs_or_logits, "logits") else outputs_or_logits
        if not torch.is_tensor(logits):
            return {}
        if self.is_regression:
            preds = logits.view(-1).float()
            targets = y.view(-1).float()
            mse = F.mse_loss(preds, targets).item()
            mae = F.l1_loss(preds, targets).item()
            out: dict[str, Any] = {"mse": float(mse), "mae": float(mae), "metric": float(-mse)}
            out.update(build_metric_payload(y_true=targets, y_pred=preds, kind="regression"))
            return out

        preds = logits.argmax(dim=-1)
        acc = (preds == y.long()).float().mean().item()
        out = {"acc": float(acc), "metric": float(acc)}
        out.update(build_metric_payload(y_true=y.long(), y_pred=preds, kind="classification"))
        return out


__all__ = ["ClassificationTask", "GLUETask"]
"""Task metadata and defaults for supervised learning problems."""
