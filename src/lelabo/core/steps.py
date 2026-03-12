# lab/core/steps.py
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, Tuple

import torch

from ..metrics.payload import build_metric_payload, is_metric_payload_key
from .batch import to_device, extract_loss_and_stats


def resolve_loss_parts(loss_or_objective: Any) -> tuple[Any, str, Any | None]:
    if callable(loss_or_objective):
        loss_fn = loss_or_objective
        loss_name = getattr(loss_or_objective, "__lelabo_loss_name__", None)
        module = getattr(loss_or_objective, "module", None)
        if not isinstance(loss_name, str) or not loss_name.strip():
            loss_name = getattr(loss_or_objective, "__name__", None)
        if not isinstance(loss_name, str) or not loss_name.strip():
            if module is not None:
                loss_name = type(module).__name__
            else:
                loss_name = type(loss_or_objective).__name__
        return loss_fn, str(loss_name).strip(), module

    loss_method = getattr(loss_or_objective, "loss", None)
    if not callable(loss_method):
        raise TypeError(
            "Expected a callable loss or an object exposing .loss(predictions, targets). "
            f"Got {type(loss_or_objective).__name__}."
        )

    inner_loss = getattr(loss_or_objective, "loss_fn", None)
    module = getattr(inner_loss, "module", None)
    loss_name = getattr(loss_or_objective, "loss_name", None)
    if not isinstance(loss_name, str) or not loss_name.strip():
        loss_name = getattr(inner_loss, "__lelabo_loss_name__", None)
    if not isinstance(loss_name, str) or not loss_name.strip():
        loss_name = getattr(inner_loss, "__name__", None)
    if not isinstance(loss_name, str) or not loss_name.strip():
        if module is not None:
            loss_name = type(module).__name__
        else:
            loss_name = type(loss_or_objective).__name__
    return loss_method, str(loss_name).strip(), module


def resolve_loss_callable(loss_or_objective: Any):
    loss_fn, _loss_name, _module = resolve_loss_parts(loss_or_objective)
    return loss_fn


def loss_display_name(loss_or_objective: Any) -> str:
    _loss_fn, loss_name, _module = resolve_loss_parts(loss_or_objective)
    return str(loss_name)


def _labels_as_class_indices(logits: torch.Tensor, labels: Any) -> torch.Tensor | None:
    if not torch.is_tensor(labels):
        return None
    if labels.dim() == 2 and int(labels.size(1)) == 1 and not torch.is_floating_point(labels):
        return labels.view(-1).long()
    if labels.dim() == 1 and labels.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
        return labels.long()
    if (
        torch.is_tensor(logits)
        and logits.dim() >= 2
        and labels.dim() == 2
        and int(labels.size(1)) == int(logits.size(-1))
        and torch.is_floating_point(labels)
    ):
        label_min = float(labels.min().item()) if labels.numel() > 0 else 0.0
        label_max = float(labels.max().item()) if labels.numel() > 0 else 0.0
        row_sums = labels.sum(dim=1)
        if (
            label_min >= -1e-6
            and label_max <= 1.0 + 1e-6
            and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
        ):
            return labels.argmax(dim=1).long()
    return None


def maybe_accuracy_from_logits(logits: torch.Tensor, labels: Any) -> float:
    if not torch.is_tensor(logits):
        return float("nan")
    if logits.dim() < 2:
        return float("nan")
    label_idx = _labels_as_class_indices(logits, labels)
    if label_idx is None:
        return float("nan")
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        return float((preds == label_idx.to(device=preds.device)).float().mean().item())


def _merge_metric_stats(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, (int, float)):
            dst[str(key)] = float(value)
            continue
        if is_metric_payload_key(str(key)):
            dst[str(key)] = value


def metric_payload_from_outputs(outputs_or_logits: Any, labels: Any) -> dict[str, Any]:
    logits = outputs_or_logits.logits if hasattr(outputs_or_logits, "logits") else outputs_or_logits
    if not torch.is_tensor(logits):
        return {}

    label_idx = _labels_as_class_indices(logits, labels)
    if label_idx is not None and logits.dim() >= 2:
        preds = logits.argmax(dim=-1)
        return build_metric_payload(y_true=label_idx.long(), y_pred=preds, kind="classification")

    if not torch.is_tensor(labels):
        return {}

    # Regression fallback: compare flattened predictions and targets.
    preds = logits.reshape(-1).to(dtype=torch.float32)
    true = labels.reshape(-1).to(dtype=torch.float32)
    return build_metric_payload(y_true=true, y_pred=preds, kind="regression")


def generic_supervised_stats(outputs_or_logits: Any, labels: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    logits = outputs_or_logits.logits if hasattr(outputs_or_logits, "logits") else outputs_or_logits
    if torch.is_tensor(logits):
        acc = maybe_accuracy_from_logits(logits, labels)
        if math.isfinite(acc):
            stats["acc"] = float(acc)
    _merge_metric_stats(stats, metric_payload_from_outputs(outputs_or_logits, labels))
    return stats


def compute_loss_and_stats(model, loss_or_objective, batch, device: str, *, hf_outputs_to_stats: bool = True):
    loss_fn = resolve_loss_callable(loss_or_objective)

    if isinstance(batch, Mapping):
        b = to_device(batch, device)
        outputs = model(**b)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            if "labels" not in b:
                raise ValueError("HF batch missing 'labels' and outputs has no .loss")
            if not hasattr(outputs, "logits"):
                raise ValueError("HF outputs has no logits; cannot compute loss.")
            loss = loss_fn(outputs.logits, b["labels"])
        if not torch.is_tensor(loss):
            raise TypeError(
                "Supervised loss must return a torch.Tensor for mapping/HuggingFace batches. "
                f"Got {type(loss).__name__}."
            )

        stats: Dict[str, Any] = {}
        if hf_outputs_to_stats:
            if hasattr(outputs, "logits") and "labels" in b:
                _merge_metric_stats(stats, generic_supervised_stats(outputs.logits, b["labels"]))
        return loss, stats

    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    x, y = to_device(batch[0], device), to_device(batch[1], device)
    out = model(x)
    loss_res = loss_fn(out, y)
    loss, stats = extract_loss_and_stats(loss_res)
    if not isinstance(stats, dict):
        stats = dict(stats) if isinstance(stats, Mapping) else {}
    if not any(is_metric_payload_key(k) for k in stats.keys()):
        _merge_metric_stats(stats, generic_supervised_stats(out, y))
    else:
        generic = generic_supervised_stats(out, y)
        if "acc" in generic and "acc" not in stats:
            stats["acc"] = float(generic["acc"])

    return loss, stats
