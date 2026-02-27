# lab/core/steps.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

import torch

from ..metrics.payload import build_metric_payload, is_metric_payload_key
from .batch import to_device, extract_loss_and_stats


def maybe_accuracy_from_logits(logits: torch.Tensor, labels: Any) -> float:
    if not torch.is_tensor(logits):
        return float("nan")
    if not torch.is_tensor(labels):
        return float("nan")
    if labels.dtype not in (torch.int64, torch.int32, torch.int16):
        return float("nan")
    if logits.dim() < 2:
        return float("nan")
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        return float((preds == labels).float().mean().item())


def _merge_metric_stats(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, (int, float)):
            dst[str(key)] = float(value)
            continue
        if is_metric_payload_key(str(key)):
            dst[str(key)] = value


def metric_payload_from_outputs(outputs_or_logits: Any, labels: Any) -> dict[str, Any]:
    if not torch.is_tensor(labels):
        return {}

    logits = outputs_or_logits.logits if hasattr(outputs_or_logits, "logits") else outputs_or_logits
    if not torch.is_tensor(logits):
        return {}

    # Classification by convention: integer labels and class logits [B, C].
    if labels.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8) and logits.dim() >= 2:
        preds = logits.argmax(dim=-1)
        return build_metric_payload(y_true=labels.long(), y_pred=preds, kind="classification")

    # Regression fallback: compare flattened predictions and targets.
    preds = logits.reshape(-1).to(dtype=torch.float32)
    true = labels.reshape(-1).to(dtype=torch.float32)
    return build_metric_payload(y_true=true, y_pred=preds, kind="regression")


def compute_loss_and_stats(model, task, batch, device: str, *, hf_outputs_to_stats: bool = True):
    if isinstance(batch, Mapping):
        b = to_device(batch, device)
        outputs = model(**b)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            if "labels" not in b:
                raise ValueError("HF batch missing 'labels' and outputs has no .loss")
            if not hasattr(outputs, "logits"):
                raise ValueError("HF outputs has no logits; can't compute task loss")
            loss = task.loss(outputs.logits, b["labels"])

        stats: Dict[str, Any] = {}
        if hf_outputs_to_stats:
            if hasattr(outputs, "logits") and "labels" in b:
                if hasattr(task, "metrics"):
                    met = task.metrics(outputs.logits, b["labels"])
                    if isinstance(met, Mapping):
                        _merge_metric_stats(stats, met)
                if "acc" not in stats:
                    stats["acc"] = maybe_accuracy_from_logits(outputs.logits, b["labels"])
                if not any(is_metric_payload_key(k) for k in stats.keys()):
                    _merge_metric_stats(stats, metric_payload_from_outputs(outputs.logits, b["labels"]))
        return loss, stats

    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    x, y = to_device(batch[0], device), to_device(batch[1], device)
    out = model(x)
    loss_res = task.loss(out, y)
    loss, stats = extract_loss_and_stats(loss_res)
    if not isinstance(stats, dict):
        stats = dict(stats) if isinstance(stats, Mapping) else {}

    if hasattr(task, "metrics"):
        met = task.metrics(out, y)
        if isinstance(met, Mapping):
            _merge_metric_stats(stats, met)
    if not any(is_metric_payload_key(k) for k in stats.keys()):
        _merge_metric_stats(stats, metric_payload_from_outputs(out, y))

    return loss, stats
