from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ..base import OptimizerUpdateRule
from ..helpers import (
    activation_derivative_from_preact,
    assign_conv2d_grads_from_activations_,
    assign_linear_grads_from_activations_,
    resolve_activation_name,
)
from ...core.batch import extract_loss_and_stats, to_device
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

    def on_train_start(self, model, task, device, state=None) -> None:
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

    def _output_delta_logits(self, task, out: Any, y: Any) -> torch.Tensor:
        if isinstance(y, Mapping):
            raise NotImplementedError("DFA v1 does not support RL/mapping labels.")
        if not hasattr(task, "output_deltas"):
            raise NotImplementedError("DFA requires task.output_deltas(out, y).")
        raw = task.output_deltas(out, y)
        if not isinstance(raw, Mapping):
            raise RuntimeError("task.output_deltas must return a dict[str, Tensor].")
        if "logits" not in raw:
            raise RuntimeError("DFA v1 expects task.output_deltas(...) to contain key 'logits'.")
        d = raw["logits"]
        if not torch.is_tensor(d):
            raise RuntimeError("DFA expects output_deltas['logits'] to be a tensor.")
        d2 = self._to_2d(d)
        if d2.numel() == 0:
            raise RuntimeError("DFA output_deltas['logits'] is empty.")
        if self.delta_scale != 1.0:
            d2 = d2 * self.delta_scale
        return d2

    @staticmethod
    def _best_effort_stats(task, out: Any, y: Any) -> dict[str, float]:
        stats: dict[str, float] = {}
        try:
            loss_res = task.loss(out, y)
            loss, extra = extract_loss_and_stats(loss_res)
            if torch.is_tensor(loss):
                stats["loss"] = float(loss.item())
            for key, value in extra.items():
                stats[str(key)] = float(value)
        except Exception:
            pass

        if hasattr(task, "metrics"):
            try:
                met = task.metrics(out, y)
                if isinstance(met, Mapping):
                    for key, value in met.items():
                        if isinstance(value, (int, float)):
                            stats[str(key)] = float(value)
            except Exception:
                pass
        return stats

    @torch.no_grad()
    def train_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        model.train()
        self.zero_grad()

        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise RuntimeError(f"DFA v1 expects batch=(x, y), got {type(batch)}.")

        if self.activation_name is None:
            self.activation_name = resolve_activation_name(model)

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)

        spec = CacheSpec(
            param_module_types=(nn.Linear, nn.Conv2d),
            require_block_inputs=True,
            require_block_outputs=True,
            require_single_call=True,
            require_single_output_head=True,
        )
        out, cache, views = forward_with_standard_cache(model, x, cache_spec=spec)
        param_blocks = views["param_blocks"]
        output_blocks = views["output_blocks"]
        if len(output_blocks) != 1:
            raise RuntimeError(f"DFA v1 expects exactly one output block, got {len(output_blocks)}.")

        output_block = output_blocks[0]
        output_name = str(getattr(output_block, "name", ""))
        output_layer = getattr(output_block, "module", None)
        if not isinstance(output_layer, nn.Linear):
            raise NotImplementedError(
                f"DFA v1 supports only Linear output head for now, got {type(output_layer)} "
                f"on block '{output_name}'."
            )

        block_inputs = cache["block_inputs"]
        block_outputs = cache["block_outputs"]

        if output_name not in block_inputs:
            raise RuntimeError(f"DFA missing cache['block_inputs'][{output_name!r}] for output block.")
        x_out = block_inputs[output_name]
        if not torch.is_tensor(x_out) or x_out.dim() != 2:
            raise RuntimeError(f"DFA output block '{output_name}' expects a 2D input tensor.")

        delta_logits = self._output_delta_logits(task, out, y)
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

        hidden_blocks = [b for b in param_blocks if not bool(getattr(b, "is_output", False))]
        for block in hidden_blocks:
            name = str(getattr(block, "name", ""))
            layer = getattr(block, "module", None)
            if not isinstance(layer, (nn.Linear, nn.Conv2d)):
                raise NotImplementedError(
                    f"DFA v1 supports hidden blocks of type Linear/Conv2d only, got {type(layer)} on '{name}'."
                )

            if name not in block_inputs:
                raise RuntimeError(f"DFA missing cache['block_inputs'][{name!r}] for hidden block.")
            if name not in block_outputs:
                raise RuntimeError(f"DFA missing cache['block_outputs'][{name!r}] for hidden block.")

            x_hidden = block_inputs[name]
            u_hidden = block_outputs[name]
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
        stats = self._best_effort_stats(task, out, y)
        self._mark_step_done()
        return stats
