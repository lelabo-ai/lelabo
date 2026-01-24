# lab/core/steps.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

import torch

try:
    from .batch import to_device, extract_loss_and_stats
except Exception:
    from batch import to_device, extract_loss_and_stats


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

        stats: Dict[str, float] = {}
        if hf_outputs_to_stats:
            try:
                if hasattr(outputs, "logits") and "labels" in b:
                    stats["acc"] = maybe_accuracy_from_logits(outputs.logits, b["labels"])
            except Exception:
                pass
        return loss, stats

    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    x, y = to_device(batch[0], device), to_device(batch[1], device)
    out = model(x)
    loss_res = task.loss(out, y)
    loss, stats = extract_loss_and_stats(loss_res)

    if hasattr(task, "metrics"):
        try:
            met = task.metrics(out, y)
            if isinstance(met, Mapping):
                for k, v in met.items():
                    if isinstance(v, (int, float)):
                        stats[k] = float(v)
        except Exception:
            pass

    return loss, stats
