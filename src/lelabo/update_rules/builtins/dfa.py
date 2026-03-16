"""Direct Feedback Alignment update rule implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ..base import OptimizerUpdateRule
from ..helpers import (
    activation_derivative_from_preact,
    normalize_activation_name,
    assign_conv2d_grads_from_activations_,
    assign_linear_grads_from_activations_,
    resolve_activation_name,
)
from ..teaching_signals import best_effort_stats, logits_delta_from_loss
from ...core.batch import to_device
from ...models.cache_provider import CacheSpec, forward_with_standard_cache


class DirectFeedbackAlignment(OptimizerUpdateRule):
    """DFA v1: supervised-only, single output head (Linear), hidden Linear/Conv2d."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        feedback_scale: float = 1.0,
        grad_clip: float | None = None,
        delta_scale: float = 1.0,
        average_grads: bool = False,
        activation_name: str | None = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            grad_clip=grad_clip,
            strict_require_grads=True,
            check_finite_grads=True,
        )
        self.feedback_scale = float(feedback_scale)
        self.delta_scale = float(delta_scale)
        self.average_grads = bool(average_grads)
        self.activation_name = None if activation_name is None else str(activation_name).lower()

        self._feedback: dict[str, torch.Tensor] = {}
        self._feedback_shapes: dict[str, tuple[int, int]] = {}

    def on_train_start(self, model, objective, device, state=None) -> None:
        _ = (objective, device, state)
        if self.activation_name is None:
            self.activation_name = resolve_activation_name(model)

    @staticmethod
    def _to_2d(t: torch.Tensor) -> torch.Tensor:
        return t if t.dim() == 2 else t.view(t.size(0), -1)

    def _ensure_feedback(self, key: str, out_dim: int, hidden_dim: int, device, dtype) -> torch.Tensor:
        shape = (int(out_dim), int(hidden_dim))
        if key in self._feedback and self._feedback_shapes.get(key) == shape:
            mat = self._feedback[key].to(device=device, dtype=dtype)
            self._feedback[key] = mat
            return mat

        std = self.feedback_scale / math.sqrt(max(1, int(out_dim)))
        mat = torch.randn(*shape, device=device, dtype=dtype) * std
        self._feedback[key] = mat
        self._feedback_shapes[key] = shape
        return mat

    def state_dict(self) -> dict[str, Any]:
        out = super().state_dict()
        out["activation_name"] = self.activation_name
        out["feedback"] = {
            str(key): value.detach().cpu().clone()
            for key, value in self._feedback.items()
            if torch.is_tensor(value)
        }
        out["feedback_shapes"] = {
            str(key): tuple(int(v) for v in shape)
            for key, shape in self._feedback_shapes.items()
        }
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "activation_name" in state and state["activation_name"] is not None:
            self.activation_name = str(state["activation_name"]).lower()

        raw_feedback = state.get("feedback", {})
        self._feedback = {}
        if isinstance(raw_feedback, Mapping):
            for key, value in raw_feedback.items():
                if torch.is_tensor(value):
                    self._feedback[str(key)] = value.detach().clone()

        raw_shapes = state.get("feedback_shapes", {})
        self._feedback_shapes = {}
        if isinstance(raw_shapes, Mapping):
            for key, shape in raw_shapes.items():
                if isinstance(shape, (tuple, list)):
                    self._feedback_shapes[str(key)] = tuple(int(v) for v in shape)
        for key, value in self._feedback.items():
            if key not in self._feedback_shapes:
                self._feedback_shapes[key] = tuple(int(v) for v in value.shape)

    def _output_delta_logits(self, objective, out: Any, y: Any) -> torch.Tensor:
        d2 = self._to_2d(
            logits_delta_from_loss(objective, out, y, rule_name="DFA")
        )
        if d2.numel() == 0:
            raise RuntimeError("DFA output_deltas['logits'] is empty.")
        if self.delta_scale != 1.0:
            d2 = d2 * self.delta_scale
        return d2

    @torch.no_grad()
    def train_step(self, model, objective, batch, device, state=None) -> dict[str, Any]:
        model.train()
        self.zero_grad()

        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise RuntimeError(f"DFA v1 expects batch=(x, y), got {type(batch)}.")

        if self.activation_name is None:
            self.activation_name = resolve_activation_name(model)
        self.activation_name = normalize_activation_name(self.activation_name)

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)

        spec = CacheSpec(
            trainable_module_types=(nn.Linear, nn.Conv2d),
            observed_module_types=(nn.Linear, nn.Conv2d),
            capture_inputs=True,
            capture_outputs=True,
            require_single_call=True,
            require_single_output_head=True,
        )
        out, _cache, views = forward_with_standard_cache(model, x, cache_spec=spec)
        execution_blocks = views.get("execution", [])
        if not isinstance(execution_blocks, list):
            raise RuntimeError("DFA v1 expects views['execution'] list.")
        output_blocks = [b for b in execution_blocks if isinstance(b, Mapping) and bool(b.get("is_output", False))]
        if len(output_blocks) != 1:
            raise RuntimeError(
                f"DFA v1 expects exactly one output block in views['execution'], got {len(output_blocks)}."
            )
        output_block = output_blocks[0]

        output_name = str(output_block.get("name", ""))
        output_layer = output_block.get("module")
        if not isinstance(output_layer, nn.Linear):
            raise NotImplementedError(
                f"DFA v1 supports only Linear output head for now, got {type(output_layer)} "
                f"on block '{output_name}'."
            )

        x_out = output_block.get("x")
        if not torch.is_tensor(x_out) or x_out.dim() != 2:
            raise RuntimeError(f"DFA output block '{output_name}' expects a 2D input tensor.")

        delta_logits = self._output_delta_logits(objective, out, y)
        if int(delta_logits.size(1)) != int(output_layer.out_features):
            raise RuntimeError(
                f"DFA logits delta shape mismatch: delta={tuple(delta_logits.shape)} "
                f"vs out_features={output_layer.out_features} for output block '{output_name}'."
            )

        assign_linear_grads_from_activations_(
            output_layer,
            x_out,
            delta_logits,
            average_batch=self.average_grads,
        )

        trainable_blocks = [b for b in execution_blocks if bool(b.get("is_trainable", False))]
        hidden_blocks = [b for b in trainable_blocks if not bool(b.get("is_output", False))]
        for block in hidden_blocks:
            name = str(block.get("name", ""))
            layer = block.get("module")
            if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                raise NotImplementedError(
                    f"DFA v1 supports hidden blocks of type Linear/Conv2d only, got {type(layer)} on '{name}'."
                )

            x_hidden = block.get("x")
            u_hidden = block.get("u")
            if not torch.is_tensor(x_hidden) or not torch.is_tensor(u_hidden):
                raise RuntimeError(f"DFA hidden block '{name}' expects tensor cache entries.")
            if int(u_hidden.size(0)) != int(delta_logits.size(0)):
                raise RuntimeError(
                    f"DFA hidden block '{name}' batch mismatch: output={int(u_hidden.size(0))}, "
                    f"logits_delta={int(delta_logits.size(0))}."
                )

            if isinstance(layer, nn.Linear):
                if x_hidden.dim() != 2 or u_hidden.dim() != 2:
                    raise RuntimeError(f"DFA hidden Linear block '{name}' expects 2D input/output tensors.")
                feedback = self._ensure_feedback(
                    key=name,
                    out_dim=int(delta_logits.size(1)),
                    hidden_dim=int(layer.out_features),
                    device=delta_logits.device,
                    dtype=delta_logits.dtype,
                )
                act_grad = activation_derivative_from_preact(self.activation_name, u_hidden)
                delta_hidden = (delta_logits @ feedback) * act_grad
                assign_linear_grads_from_activations_(
                    layer,
                    x_hidden,
                    delta_hidden,
                    average_batch=self.average_grads,
                )
                continue

            if x_hidden.dim() != 4 or u_hidden.dim() != 4:
                raise RuntimeError(f"DFA hidden Conv2d block '{name}' expects 4D input/output tensors.")
            hidden_dim = int(u_hidden[0].numel())
            feedback = self._ensure_feedback(
                key=name,
                out_dim=int(delta_logits.size(1)),
                hidden_dim=hidden_dim,
                device=delta_logits.device,
                dtype=delta_logits.dtype,
            )
            act_grad = activation_derivative_from_preact(self.activation_name, u_hidden)
            delta_hidden = (delta_logits @ feedback).view_as(u_hidden) * act_grad
            assign_conv2d_grads_from_activations_(
                layer,
                x_hidden,
                delta_hidden,
                average_batch=self.average_grads,
            )

        self.step(model.parameters(), require_grads=True, check_finite_grads=True)
        stats = best_effort_stats(objective, out, y)
        self._mark_step_done()
        return stats
"""Direct Feedback Alignment update rule implementation."""
