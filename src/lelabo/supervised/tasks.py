from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from ..metrics.payload import build_metric_payload


def _onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(labels.size(0), num_classes, device=labels.device, dtype=torch.float32)
    out.scatter_(1, labels.view(-1, 1), 1.0)
    return out


def _classification_targets(y: torch.Tensor, *, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    if y.dim() == 2 and int(y.size(1)) == 1 and not torch.is_floating_point(y):
        y = y.view(-1)
    if y.dim() == 1:
        return _onehot(y.long(), num_classes=num_classes).to(dtype=dtype)
    if y.dim() == 2 and int(y.size(1)) == int(num_classes):
        return y.to(dtype=dtype)
    raise ValueError(
        "Expected class targets shaped [B], [B,1], or [B,C]; "
        f"got {tuple(y.shape)} with C={int(num_classes)}."
    )


def _as_class_indices(y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 2 and int(y.size(1)) == 1 and not torch.is_floating_point(y):
        return y.view(-1).long()
    if y.dim() == 2 and torch.is_floating_point(y):
        return y.argmax(dim=1).long()
    if y.dim() == 1:
        return y.long()
    raise ValueError(f"Expected labels shaped [B], [B,1], or one-hot [B,C], got {tuple(y.shape)}.")


def _loss_reduction(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None) -> str:
    module = getattr(loss_fn, "module", None)
    reduction = getattr(module, "reduction", "mean")
    return str(reduction).strip().lower()


def _apply_reduction_to_delta(delta: torch.Tensor, reduction: str) -> torch.Tensor:
    red = str(reduction).strip().lower()
    if red == "mean":
        denom = max(1, int(delta.numel()))
        return delta / float(denom)
    if red in {"sum", "none"}:
        return delta
    raise ValueError(f"Unsupported loss reduction '{reduction}'.")


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

    @torch.no_grad()
    def output_deltas(self, logits: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        name = self.loss_name
        reduction = _loss_reduction(self.loss_fn)
        num_classes = int(logits.size(1))
        targets = _classification_targets(y, num_classes=num_classes, dtype=logits.dtype).to(device=logits.device)

        if name in {"cross_entropy", "ce"}:
            module = getattr(self.loss_fn, "module", None)
            if module is not None:
                weight = getattr(module, "weight", None)
                ignore_index = int(getattr(module, "ignore_index", -100))
                if weight is not None or ignore_index != -100:
                    raise NotImplementedError(
                        "output_deltas for cross_entropy supports only default weight/ignore_index."
                    )
                smooth = float(getattr(module, "label_smoothing", 0.0))
            else:
                smooth = 0.0
            if smooth > 0.0:
                targets = targets * (1.0 - smooth) + (smooth / float(max(1, num_classes)))
            delta = torch.softmax(logits, dim=1) - targets
            batch = max(1, int(logits.size(0)))
            if reduction == "mean":
                delta = delta / float(batch)
            elif reduction not in {"sum", "none"}:
                raise ValueError(f"Unsupported cross-entropy reduction '{reduction}'.")
            return {"logits": delta}

        if name in {"bce_with_logits", "bce_logits"}:
            module = getattr(self.loss_fn, "module", None)
            if module is not None and (
                getattr(module, "weight", None) is not None
                or getattr(module, "pos_weight", None) is not None
            ):
                raise NotImplementedError(
                    "output_deltas for bce_with_logits supports only default weight/pos_weight."
                )
            delta = torch.sigmoid(logits) - targets
            return {"logits": _apply_reduction_to_delta(delta, reduction)}

        if name in {"bce", "binary_cross_entropy"}:
            module = getattr(self.loss_fn, "module", None)
            if module is not None and getattr(module, "weight", None) is not None:
                raise NotImplementedError("output_deltas for bce supports only default weights.")
            eps = torch.finfo(logits.dtype).eps
            probs = logits.clamp(min=eps, max=1.0 - eps)
            delta = (probs - targets) / (probs * (1.0 - probs))
            return {"logits": _apply_reduction_to_delta(delta, reduction)}

        if name in {"mse", "mse_loss"}:
            if torch.is_tensor(y) and y.dim() == logits.dim():
                target = y.to(device=logits.device, dtype=logits.dtype)
            else:
                target = targets
            delta = 2.0 * (logits - target)
            return {"logits": _apply_reduction_to_delta(delta, reduction)}

        raise NotImplementedError(f"output_deltas is not implemented for loss '{self.loss_name}'.")


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

    @torch.no_grad()
    def evaluate(self, model, loader, device: str) -> dict[str, float]:
        total_n = 0
        total_loss = 0.0
        total_acc = 0.0
        total_mse = 0.0
        total_mae = 0.0

        for batch in loader:
            if not isinstance(batch, dict):
                raise TypeError("GLUETask expects mapping batches with HuggingFace collate output.")

            feed: dict[str, Any] = {}
            for key, value in batch.items():
                feed[key] = value.to(device) if torch.is_tensor(value) else value

            outputs = model(**feed)
            labels = feed["labels"]
            logits = outputs.logits
            loss = outputs.loss if getattr(outputs, "loss", None) is not None else self.loss(logits, labels)

            n = int(labels.shape[0]) if hasattr(labels, "shape") and labels.shape else 1
            weight = float(max(1, n))
            total_n += int(weight)
            total_loss += float(loss.item()) * weight

            stats = self.metrics(outputs, labels)
            if self.is_regression:
                total_mse += float(stats.get("mse", 0.0)) * weight
                total_mae += float(stats.get("mae", 0.0)) * weight
            else:
                total_acc += float(stats.get("acc", 0.0)) * weight

        denom = float(max(1, total_n))
        out: dict[str, float] = {"loss": float(total_loss / denom)}
        if self.is_regression:
            mse = float(total_mse / denom)
            out["mse"] = mse
            out["mae"] = float(total_mae / denom)
            out["metric"] = float(-mse)
        else:
            acc = float(total_acc / denom)
            out["acc"] = acc
            out["metric"] = acc
        return out


__all__ = ["ClassificationTask", "GLUETask"]
