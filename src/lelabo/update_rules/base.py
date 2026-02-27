from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Dict

import torch


class UpdateRule(ABC):
    """Common contract for all update rules."""

    def __init__(self) -> None:
        self.global_step = 0

    def on_train_start(self, model, task, device, state=None) -> None:
        pass

    @abstractmethod
    def train_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        raise NotImplementedError

    @torch.no_grad()
    def on_eval_start(self, model, task, device, state=None) -> None:
        model.eval()

    def state_dict(self) -> Dict[str, Any]:
        return {"global_step": int(self.global_step)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "global_step" in state:
            self.global_step = int(state["global_step"])

    def _mark_step_done(self) -> None:
        self.global_step += 1


class OptimizerUpdateRule(UpdateRule):
    """Base class for update rules driven by a torch optimizer."""

    def __init__(self, optimizer: torch.optim.Optimizer, grad_clip: float | None = None) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _optimizer_params(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for group in self.optimizer.param_groups:
            for p in group.get("params", []):
                if isinstance(p, torch.nn.Parameter):
                    params.append(p)
        return params

    def step(self, params_for_clip: Iterable[torch.nn.Parameter] | None = None) -> None:
        if self.grad_clip is not None:
            params = list(params_for_clip) if params_for_clip is not None else self._optimizer_params()
            if params:
                torch.nn.utils.clip_grad_norm_(params, float(self.grad_clip))
        self.optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        out = super().state_dict()
        out["optimizer"] = self.optimizer.state_dict()
        out["grad_clip"] = self.grad_clip
        return out

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "grad_clip" in state:
            self.grad_clip = state["grad_clip"]
