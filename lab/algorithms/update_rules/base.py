# lab/algorithms/update_rules/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Dict, Optional

import torch

from .helpers import to_device, extract_loss_and_stats


class UpdateRule(ABC):
    def __init__(self):
        self.global_step = 0

    def on_train_start(self, model, task, device):
        pass

    @abstractmethod
    def train_step(self, model, task, batch, device) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def on_eval_start(self, model, task, device):
        model.eval()


class AutogradUpdateRule(UpdateRule):
    """
    Base pour toutes les règles qui font:
      forward -> loss -> backward -> clip -> optimizer.step
    """
    def __init__(self, optimizer: torch.optim.Optimizer, grad_clip: Optional[float] = None):
        super().__init__()
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def compute_loss(self, model, task, batch, device):
        # HF dict batch
        if isinstance(batch, Mapping):
            batch = to_device(batch, device)
            outputs = model(**batch)
            return outputs.loss, {"acc": _maybe_acc(outputs, batch)}

        # tuple batch
        x, y = to_device(batch, device)
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

    def train_step(self, model, task, batch, device) -> Dict[str, float]:
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss, stats = self.compute_loss(model, task, batch, device)
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        self.optimizer.step()

        out = dict(stats)
        out["loss"] = float(loss.item())
        self.global_step += 1
        return out


def _maybe_acc(outputs, batch) -> float:
    if not hasattr(outputs, "logits"):
        return float("nan")
    if "labels" not in batch:
        return float("nan")
    labels = batch["labels"]
    if not torch.is_tensor(labels):
        return float("nan")
    if labels.dtype not in (torch.int64, torch.int32, torch.int16):
        return float("nan")
    with torch.no_grad():
        preds = outputs.logits.argmax(dim=-1)
        return float((preds == labels).float().mean().item())
