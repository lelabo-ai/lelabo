"""Teaching-signal helpers used by feedback-based update rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..core.batch import extract_loss_and_stats
from ..core.steps import generic_supervised_stats, resolve_loss_parts


def _classification_targets(y: torch.Tensor, *, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    if y.dim() == 2 and int(y.size(1)) == 1 and not torch.is_floating_point(y):
        y = y.view(-1)
    if y.dim() == 1:
        out = torch.zeros(y.size(0), num_classes, device=y.device, dtype=torch.float32)
        out.scatter_(1, y.long().view(-1, 1), 1.0)
        return out.to(dtype=dtype)
    if y.dim() == 2 and int(y.size(1)) == int(num_classes):
        return y.to(dtype=dtype)
    raise ValueError(
        "Expected class targets shaped [B], [B,1], or [B,C]; "
        f"got {tuple(y.shape)} with C={int(num_classes)}."
    )


def _loss_reduction(module: Any | None) -> str:
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


def logits_from_outputs(outputs_or_logits: Any, *, rule_name: str) -> torch.Tensor:
    if isinstance(outputs_or_logits, Mapping):
        if "logits" in outputs_or_logits and torch.is_tensor(outputs_or_logits["logits"]):
            return outputs_or_logits["logits"]
        raise RuntimeError(f"{rule_name} expects mapping outputs to contain tensor key 'logits'.")
    if hasattr(outputs_or_logits, "logits") and torch.is_tensor(outputs_or_logits.logits):
        return outputs_or_logits.logits
    if not torch.is_tensor(outputs_or_logits):
        raise RuntimeError(
            f"{rule_name} expects tensor outputs (or mapping/.logits), got {type(outputs_or_logits)}."
        )
    return outputs_or_logits


def infer_num_classes(y: torch.Tensor, logits: torch.Tensor) -> int:
    if logits.dim() >= 2:
        return int(logits.size(-1))
    if y.dim() == 2:
        return int(y.size(1))
    return int(y.max().item()) + 1


def _supported_loss_names() -> tuple[str, ...]:
    return ("ce", "cross_entropy", "bce_logits", "bce_with_logits", "bce", "binary_cross_entropy", "mse", "mse_loss")


def logits_delta_from_loss(loss_or_objective: Any, outputs_or_logits: Any, y: Any, *, rule_name: str) -> torch.Tensor:
    if isinstance(y, Mapping):
        raise NotImplementedError(f"{rule_name} does not support mapping labels/targets.")
    if not torch.is_tensor(y):
        raise RuntimeError(f"{rule_name} expects tensor labels/targets, got {type(y)}.")

    logits = logits_from_outputs(outputs_or_logits, rule_name=rule_name)
    if logits.numel() == 0:
        raise RuntimeError(f"{rule_name} received an empty logits tensor.")

    loss_fn, loss_name, module = resolve_loss_parts(loss_or_objective)
    name = str(loss_name).strip().lower()
    reduction = _loss_reduction(module)
    num_classes = infer_num_classes(y, logits)
    targets = _classification_targets(y.to(device=logits.device), num_classes=num_classes, dtype=logits.dtype)

    if name in {"ce", "cross_entropy"}:
        if module is not None:
            weight = getattr(module, "weight", None)
            ignore_index = int(getattr(module, "ignore_index", -100))
            if weight is not None or ignore_index != -100:
                raise NotImplementedError(
                    f"{rule_name} supports cross_entropy only with default weight/ignore_index."
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
        return delta

    if name in {"bce_with_logits", "bce_logits"}:
        if module is not None and (
            getattr(module, "weight", None) is not None
            or getattr(module, "pos_weight", None) is not None
        ):
            raise NotImplementedError(
                f"{rule_name} supports bce_with_logits only with default weight/pos_weight."
            )
        delta = torch.sigmoid(logits) - targets
        return _apply_reduction_to_delta(delta, reduction)

    if name in {"bce", "binary_cross_entropy"}:
        if module is not None and getattr(module, "weight", None) is not None:
            raise NotImplementedError(f"{rule_name} supports bce only with default weights.")
        eps = torch.finfo(logits.dtype).eps
        probs = logits.clamp(min=eps, max=1.0 - eps)
        delta = (probs - targets) / (probs * (1.0 - probs))
        return _apply_reduction_to_delta(delta, reduction)

    if name in {"mse", "mse_loss"}:
        if torch.is_tensor(y) and y.dim() == logits.dim():
            target = y.to(device=logits.device, dtype=logits.dtype)
        else:
            target = targets
        delta = 2.0 * (logits - target)
        return _apply_reduction_to_delta(delta, reduction)

    supported = ", ".join(_supported_loss_names())
    raise NotImplementedError(
        f"{rule_name} does not support loss '{loss_name}'. Supported losses: {supported}."
    )


def best_effort_stats(loss_or_objective: Any, outputs_or_logits: Any, y: Any) -> dict[str, float]:
    stats: dict[str, float] = {}
    if torch.is_tensor(y):
        try:
            loss_fn, _loss_name, _module = resolve_loss_parts(loss_or_objective)
            loss_res = loss_fn(outputs_or_logits, y)
            loss, extra = extract_loss_and_stats(loss_res)
            if torch.is_tensor(loss):
                stats["loss"] = float(loss.item())
            for key, value in extra.items():
                stats[str(key)] = float(value)
        except Exception:
            pass

        try:
            generic = generic_supervised_stats(outputs_or_logits, y)
            for key, value in generic.items():
                if isinstance(value, (int, float)):
                    stats[str(key)] = float(value)
        except Exception:
            pass
    return stats
"""Teaching-signal helpers used by feedback-based update rules."""
