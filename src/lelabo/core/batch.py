"""Batch normalization helpers used by the trainer and supervised step logic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

import torch


def to_device(obj: Any, device: str):
    """Recursively move tensors contained in `obj` to `device`."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(to_device(x, device) for x in obj)
    return obj


def is_mapping_batch(batch: Any) -> bool:
    """Return whether ``batch`` uses mapping/HuggingFace-style semantics."""
    return isinstance(batch, Mapping)


def unpack_batch(batch: Any, device: str) -> Tuple[str, Tuple[Any, ...]]:
    """Standardise batch formats.

    Returns:
      - mode: "mapping" or "tuple"
      - payload:
          mapping: (batch_dict,)
          tuple: (x, y)

    Note: the returned objects are already moved to `device`.
    """
    if is_mapping_batch(batch):
        b = to_device(batch, device)
        return "mapping", (b,)
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    x, y = batch
    return "tuple", (to_device(x, device), to_device(y, device))


def infer_batch_size(batch: Any) -> int:
    """Best-effort batch-size inference for logging/averaging."""
    if torch.is_tensor(batch):
        return int(batch.size(0)) if batch.dim() >= 1 else 1
    if isinstance(batch, Mapping):
        if "labels" in batch and torch.is_tensor(batch["labels"]):
            return int(batch["labels"].size(0)) if batch["labels"].dim() >= 1 else 1
        for v in batch.values():
            if torch.is_tensor(v):
                return int(v.size(0)) if v.dim() >= 1 else 1
        return 0
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x = batch[0]
        if torch.is_tensor(x):
            return int(x.size(0)) if x.dim() >= 1 else 1
    return 0


def extract_loss_and_stats(res) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalise different (loss, stats) conventions."""
    if torch.is_tensor(res):
        return res, {}

    if isinstance(res, tuple) and len(res) == 2:
        loss, stats = res
        if not torch.is_tensor(loss):
            raise TypeError("loss must be a torch.Tensor")
        out: Dict[str, float] = {}
        if isinstance(stats, Mapping):
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    out[k] = float(v)
        return loss, out

    if isinstance(res, Mapping):
        if "loss" not in res:
            raise ValueError("dict result must contain key 'loss'")
        loss = res["loss"]
        if not torch.is_tensor(loss):
            loss = torch.tensor(float(loss))
        out: Dict[str, float] = {}
        for k, v in res.items():
            if k == "loss":
                continue
            if isinstance(v, (int, float)):
                out[k] = float(v)
        return loss, out

    raise TypeError(f"Unsupported loss return type: {type(res)}")
