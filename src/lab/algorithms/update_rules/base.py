# lab/algorithms/update_rules/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

import torch

try:
    from ...core.steps import compute_loss_and_stats
except Exception:
    from lab.core.steps import compute_loss_and_stats


class UpdateRule(ABC):
    def __init__(self):
        self.global_step = 0

    def on_train_start(self, model, task, device, state=None):
        pass

    @abstractmethod
    def train_step(self, model, task, batch, device, state=None) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def on_eval_start(self, model, task, device, state=None):
        model.eval()

    def state_dict(self) -> Dict[str, Any]:
        return {"global_step": int(self.global_step)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "global_step" in state:
            self.global_step = int(state["global_step"])


class AutogradUpdateRule(UpdateRule):
    def __init__(self, optimizer: torch.optim.Optimizer, grad_clip: Optional[float] = None):
        super().__init__()
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def train_step(self, model, task, batch, device, state=None) -> Dict[str, float]:
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss, stats = compute_loss_and_stats(model, task, batch, device)
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        self.optimizer.step()

        out = dict(stats)
        out["loss"] = float(loss.item())
        self.global_step += 1
        return out

    def state_dict(self) -> Dict[str, Any]:
        d = super().state_dict()
        try:
            d["optimizer"] = self.optimizer.state_dict()
        except Exception:
            pass
        d["grad_clip"] = self.grad_clip
        return d

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        if "grad_clip" in state:
            self.grad_clip = state["grad_clip"]
