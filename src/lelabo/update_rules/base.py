from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Dict

import torch


class UpdateRule(ABC):
    """Common contract for all update rules."""

    def __init__(self) -> None:
        self.global_step = 0

    def on_train_start(self, model, objective, device, state=None) -> None:
        pass

    @abstractmethod
    def train_step(self, model, objective, batch, device, state=None) -> dict[str, Any]:
        raise NotImplementedError

    @torch.no_grad()
    def on_eval_start(self, model, objective, device, state=None) -> None:
        _ = objective
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

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        grad_clip: float | None = None,
        *,
        strict_require_grads: bool = False,
        check_finite_grads: bool = False,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.strict_require_grads = bool(strict_require_grads)
        self.check_finite_grads = bool(check_finite_grads)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _optimizer_params(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for group in self.optimizer.param_groups:
            for p in group.get("params", []):
                if isinstance(p, torch.nn.Parameter):
                    params.append(p)
        return params

    @staticmethod
    def _has_any_grad(params: Iterable[torch.nn.Parameter]) -> bool:
        return any(isinstance(p, torch.nn.Parameter) and p.grad is not None for p in params)

    @staticmethod
    def _assert_finite_grads(params: Iterable[torch.nn.Parameter]) -> None:
        for p in params:
            if not isinstance(p, torch.nn.Parameter):
                continue
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                raise RuntimeError("Non-finite gradients detected (NaN or Inf).")

    def step(
        self,
        params_for_clip: Iterable[torch.nn.Parameter] | None = None,
        *,
        require_grads: bool | None = None,
        check_finite_grads: bool | None = None,
    ) -> None:
        params = list(params_for_clip) if params_for_clip is not None else self._optimizer_params()

        must_have_grads = self.strict_require_grads if require_grads is None else bool(require_grads)
        must_be_finite = self.check_finite_grads if check_finite_grads is None else bool(check_finite_grads)

        if must_have_grads and not self._has_any_grad(params):
            raise RuntimeError("OptimizerUpdateRule.step() called with no gradients set.")
        if must_be_finite:
            self._assert_finite_grads(params)

        if self.grad_clip is not None:
            if params:
                torch.nn.utils.clip_grad_norm_(params, float(self.grad_clip))
        self.optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        out = super().state_dict()
        out["optimizer"] = self.optimizer.state_dict()
        out["grad_clip"] = self.grad_clip
        out["strict_require_grads"] = self.strict_require_grads
        out["check_finite_grads"] = self.check_finite_grads
        return out

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "grad_clip" in state:
            self.grad_clip = state["grad_clip"]
        if "strict_require_grads" in state:
            self.strict_require_grads = bool(state["strict_require_grads"])
        if "check_finite_grads" in state:
            self.check_finite_grads = bool(state["check_finite_grads"])
