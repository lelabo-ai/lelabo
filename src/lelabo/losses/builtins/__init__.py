"""Built-in supervised losses and target-normalization helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from ..registry import LossContext, register_loss


def _classification_targets(y: Any, *, num_classes: int, device, dtype) -> torch.Tensor:
    if not torch.is_tensor(y):
        raise TypeError(f"Expected tensor targets, got {type(y)}")

    y_t = y.to(device=device)
    if y_t.dim() == 2 and int(y_t.size(1)) == 1 and not torch.is_floating_point(y_t):
        y_t = y_t.view(-1)

    if y_t.dim() == 1:
        labels = y_t.long().view(-1)
        if labels.numel() == 0:
            raise ValueError("Empty target tensor.")
        if int(labels.min().item()) < 0 or int(labels.max().item()) >= int(num_classes):
            raise ValueError(
                f"Class labels out of range for num_classes={int(num_classes)}: "
                f"min={int(labels.min().item())}, max={int(labels.max().item())}."
            )
        out = torch.zeros(labels.size(0), int(num_classes), device=device, dtype=dtype)
        out.scatter_(1, labels.unsqueeze(1), 1.0)
        return out

    if y_t.dim() == 2:
        if int(y_t.size(1)) != int(num_classes):
            raise ValueError(
                f"Target width mismatch: got {int(y_t.size(1))}, expected {int(num_classes)}."
            )
        return y_t.to(dtype=dtype)

    raise ValueError(f"Unsupported target shape for classification loss: {tuple(y_t.shape)}")


def _as_class_indices(y: Any) -> torch.Tensor:
    if not torch.is_tensor(y):
        raise TypeError(f"Expected tensor targets, got {type(y)}")
    y_t = y
    if y_t.dim() == 2 and int(y_t.size(1)) == 1 and not torch.is_floating_point(y_t):
        y_t = y_t.view(-1)
    if y_t.dim() == 2 and torch.is_floating_point(y_t):
        return y_t.argmax(dim=1).long()
    if y_t.dim() != 1:
        raise ValueError(f"CrossEntropy expects class indices [B] or one-hot [B,C], got {tuple(y_t.shape)}")
    return y_t.long()


def _wrap_module_loss(module: nn.Module, fn: Callable[[torch.Tensor, Any], torch.Tensor]) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    # Keep a hard ref to avoid module GC and to make repr/debugging easier.
    _module = module

    def _loss(pred: torch.Tensor, target: Any) -> torch.Tensor:
        return fn(pred, target)

    setattr(_loss, "module", _module)
    return _loss


@register_loss("cross_entropy")
@register_loss("ce")
def build_cross_entropy(ctx: LossContext):
    params = ctx.loss_params()
    mod = nn.CrossEntropyLoss(**params)

    def _fn(pred: torch.Tensor, target: Any) -> torch.Tensor:
        return mod(pred, _as_class_indices(target).to(device=pred.device))

    return _wrap_module_loss(mod, _fn)


@register_loss("bce_with_logits")
@register_loss("bce_logits")
def build_bce_with_logits(ctx: LossContext):
    params = ctx.loss_params()
    mod = nn.BCEWithLogitsLoss(**params)

    def _fn(pred: torch.Tensor, target: Any) -> torch.Tensor:
        if pred.dim() != 2:
            raise ValueError(f"BCEWithLogits expects [B,C] predictions, got {tuple(pred.shape)}")
        t = _classification_targets(
            target,
            num_classes=int(pred.size(1)),
            device=pred.device,
            dtype=pred.dtype,
        )
        return mod(pred, t)

    return _wrap_module_loss(mod, _fn)


@register_loss("binary_cross_entropy")
@register_loss("bce")
def build_bce(ctx: LossContext):
    params = ctx.loss_params()
    mod = nn.BCELoss(**params)

    def _fn(pred: torch.Tensor, target: Any) -> torch.Tensor:
        if pred.dim() != 2:
            raise ValueError(f"BCE expects [B,C] predictions, got {tuple(pred.shape)}")
        t = _classification_targets(
            target,
            num_classes=int(pred.size(1)),
            device=pred.device,
            dtype=pred.dtype,
        )
        return mod(pred, t)

    return _wrap_module_loss(mod, _fn)


@register_loss("mse_loss")
@register_loss("mse")
def build_mse(ctx: LossContext):
    params = ctx.loss_params()
    mod = nn.MSELoss(**params)

    def _fn(pred: torch.Tensor, target: Any) -> torch.Tensor:
        if torch.is_tensor(target) and target.dim() == pred.dim():
            t = target.to(device=pred.device, dtype=pred.dtype)
        elif torch.is_tensor(target) and pred.dim() == 2 and int(pred.size(1)) == 1 and target.dim() == 1:
            t = target.to(device=pred.device, dtype=pred.dtype).view(-1, 1)
        elif pred.dim() == 2:
            t = _classification_targets(
                target,
                num_classes=int(pred.size(1)),
                device=pred.device,
                dtype=pred.dtype,
            )
        else:
            raise ValueError(
                f"MSE target shape mismatch: pred={tuple(pred.shape)}; provide matching tensor targets."
            )
        return mod(pred, t)

    return _wrap_module_loss(mod, _fn)
