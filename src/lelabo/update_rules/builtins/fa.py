from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class FeedbackAlignment(OptimizerUpdateRule):
    """Feedback Alignment (FA): BP on output head + random feedback through next layers."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        feedback_scale: float = 1.0,
        grad_clip: float | None = None,
        delta_scale: float = 1.0,
        average_grads: bool = False,
        activation_name: str | None = None,
        pool_backscale: bool = True,
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
        self.pool_backscale = bool(pool_backscale)
        self.activation_name = None if activation_name is None else str(activation_name).lower()

        self._feedback: dict[str, torch.Tensor] = {}
        self._feedback_shapes: dict[str, tuple[int, ...]] = {}

    def on_train_start(self, model, objective, device, state=None) -> None:
        _ = (objective, device, state)
        if self.activation_name is None:
            self.activation_name = resolve_activation_name(model)

    @staticmethod
    def _to_2d(t: torch.Tensor) -> torch.Tensor:
        return t if t.dim() == 2 else t.view(t.size(0), -1)

    def _ensure_feedback(self, key: str, next_layer: nn.Module, device, dtype) -> torch.Tensor:
        if not hasattr(next_layer, "weight"):
            raise TypeError(f"FA next layer '{key}' has no weight tensor.")
        weight = getattr(next_layer, "weight")
        if not torch.is_tensor(weight):
            raise TypeError(f"FA next layer '{key}' weight is not a tensor.")

        shape = tuple(int(v) for v in weight.shape)
        if key in self._feedback and self._feedback_shapes.get(key) == shape:
            fb = self._feedback[key].to(device=device, dtype=dtype)
            self._feedback[key] = fb
            return fb

        fan_in = max(1, int(weight[0].numel())) if int(weight.ndim) > 1 else max(1, int(weight.shape[0]))
        std = self.feedback_scale / math.sqrt(float(fan_in))
        fb = torch.randn(*shape, device=device, dtype=dtype) * std
        self._feedback[key] = fb
        self._feedback_shapes[key] = shape
        return fb

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
            logits_delta_from_loss(objective, out, y, rule_name="FA")
        )
        if d2.numel() == 0:
            raise RuntimeError("FA output_deltas['logits'] is empty.")
        if self.delta_scale != 1.0:
            d2 = d2 * self.delta_scale
        return d2

    @staticmethod
    def _align_spatial(delta: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        if delta.dim() != 4:
            raise RuntimeError(f"Expected a 4D delta tensor, got shape={tuple(delta.shape)}.")
        if int(delta.size(-2)) == int(height) and int(delta.size(-1)) == int(width):
            return delta
        return F.interpolate(delta, size=(int(height), int(width)), mode="nearest")

    def _lift_linear_feedback_to_conv(
        self,
        delta_in: torch.Tensor,
        u_current: torch.Tensor,
    ) -> torch.Tensor:
        if delta_in.dim() != 2 or u_current.dim() != 4:
            raise RuntimeError(
                "FA expected 2D linear feedback and 4D current pre-activation for Conv2d mapping."
            )
        batch_size, channels, height, width = (int(v) for v in u_current.shape)
        width_in = int(delta_in.size(1))

        if width_in == channels:
            delta4 = delta_in.view(batch_size, channels, 1, 1).expand(-1, -1, height, width)
            if self.pool_backscale:
                delta4 = delta4 / float(max(1, height * width))
            return delta4

        if width_in == channels * height * width:
            return delta_in.view(batch_size, channels, height, width)

        if width_in % channels != 0:
            raise RuntimeError(
                "FA cannot map linear feedback to conv shape: "
                f"delta_width={width_in}, channels={channels}, target_hw=({height}, {width})."
            )

        area = width_in // channels
        side = int(math.sqrt(float(area)))
        if side * side != area:
            raise RuntimeError(
                "FA cannot infer pooled spatial shape from linear feedback width. "
                f"delta_width={width_in}, channels={channels}."
            )
        delta4 = delta_in.view(batch_size, channels, side, side)
        return self._align_spatial(delta4, height=height, width=width)

    @torch.no_grad()
    def train_step(self, model, objective, batch, device, state=None) -> dict[str, Any]:
        model.train()
        self.zero_grad()

        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise RuntimeError(f"FA v1 expects batch=(x, y), got {type(batch)}.")

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

        execution_blocks = views.get("execution_blocks", [])
        output_blocks = views.get("output_blocks", [])
        if not isinstance(execution_blocks, list):
            raise RuntimeError("FA v1 expects views['execution_blocks'] list.")
        if not isinstance(output_blocks, list):
            output_blocks = []
        output_blocks = [b for b in output_blocks if isinstance(b, Mapping)]
        if len(output_blocks) != 1:
            raise RuntimeError(
                f"FA v1 expects exactly one output block in views['output_blocks'], got {len(output_blocks)}."
            )
        output_block = output_blocks[0]

        param_blocks = [b for b in execution_blocks if bool(b.get("is_trainable", False))]
        if not param_blocks:
            raise RuntimeError("FA v1 found no parametric blocks in cache views.")

        by_name = {str(b.get("name", "")): b for b in param_blocks}

        output_name = str(output_block.get("name", ""))
        output_layer = output_block.get("module")
        if not isinstance(output_layer, (nn.Linear, nn.Conv2d)):
            raise NotImplementedError(
                f"FA v1 supports only Linear/Conv2d output heads, got {type(output_layer)} on '{output_name}'."
            )

        x_out = output_block.get("x")
        u_out = output_block.get("u")
        if not torch.is_tensor(x_out) or not torch.is_tensor(u_out):
            raise RuntimeError(f"FA output block '{output_name}' expects tensor cache entries.")

        delta_next: torch.Tensor
        delta_logits = self._output_delta_logits(objective, out, y)
        if int(delta_logits.size(0)) != int(u_out.size(0)):
            raise RuntimeError(
                f"FA output batch mismatch on '{output_name}': "
                f"delta={int(delta_logits.size(0))}, output={int(u_out.size(0))}."
            )

        if isinstance(output_layer, nn.Linear):
            if x_out.dim() != 2 or u_out.dim() != 2:
                raise RuntimeError(f"FA output Linear block '{output_name}' expects 2D input/output tensors.")
            if int(delta_logits.size(1)) != int(output_layer.out_features):
                raise RuntimeError(
                    f"FA logits delta shape mismatch: delta={tuple(delta_logits.shape)} "
                    f"vs out_features={output_layer.out_features} for output block '{output_name}'."
                )
            assign_linear_grads_from_activations_(
                output_layer,
                x_out,
                delta_logits,
                average_batch=self.average_grads,
            )
            delta_next = delta_logits
        else:
            if x_out.dim() != 4 or u_out.dim() != 4:
                raise RuntimeError(f"FA output Conv2d block '{output_name}' expects 4D input/output tensors.")
            delta_next = self._lift_linear_feedback_to_conv(delta_logits, u_out)
            assign_conv2d_grads_from_activations_(
                output_layer,
                x_out,
                delta_next,
                average_batch=self.average_grads,
            )

        for idx in range(len(param_blocks) - 2, -1, -1):
            current = param_blocks[idx]
            nxt = param_blocks[idx + 1]
            current_name = str(current.get("name", ""))
            next_name = str(nxt.get("name", ""))
            current_layer = current.get("module")
            next_layer = nxt.get("module")

            if not isinstance(current_layer, (nn.Linear, nn.Conv2d)):
                raise NotImplementedError(
                    f"FA v1 supports hidden Linear/Conv2d only, got {type(current_layer)} on '{current_name}'."
                )
            if not isinstance(next_layer, (nn.Linear, nn.Conv2d)):
                raise NotImplementedError(
                    f"FA v1 supports next-layer Linear/Conv2d only, got {type(next_layer)} on '{next_name}'."
                )

            x_current = current.get("x")
            u_current = current.get("u")
            if not torch.is_tensor(x_current) or not torch.is_tensor(u_current):
                raise RuntimeError(f"FA hidden block '{current_name}' expects tensor cache entries.")

            feedback = self._ensure_feedback(
                key=next_name,
                next_layer=next_layer,
                device=u_current.device,
                dtype=u_current.dtype,
            )

            if isinstance(next_layer, nn.Linear):
                if delta_next.dim() != 2:
                    raise RuntimeError(
                        f"FA expected 2D delta for next Linear block '{next_name}', got {tuple(delta_next.shape)}."
                    )
                delta_in = delta_next @ feedback
                act_grad = activation_derivative_from_preact(self.activation_name, u_current)

                if isinstance(current_layer, nn.Linear):
                    if x_current.dim() != 2 or u_current.dim() != 2:
                        raise RuntimeError(
                            f"FA hidden Linear block '{current_name}' expects 2D input/output tensors."
                        )
                    delta_current = delta_in * act_grad
                    assign_linear_grads_from_activations_(
                        current_layer,
                        x_current,
                        delta_current,
                        average_batch=self.average_grads,
                    )
                else:
                    if x_current.dim() != 4 or u_current.dim() != 4:
                        raise RuntimeError(f"FA hidden Conv2d block '{current_name}' expects 4D tensors.")
                    delta_projected = self._lift_linear_feedback_to_conv(delta_in, u_current)
                    delta_current = delta_projected * act_grad
                    assign_conv2d_grads_from_activations_(
                        current_layer,
                        x_current,
                        delta_current,
                        average_batch=self.average_grads,
                    )
                delta_next = delta_current
                continue

            next_entry = by_name.get(next_name, {})
            x_next = next_entry.get("x")
            u_next = next_entry.get("u")
            if not torch.is_tensor(x_next) or not torch.is_tensor(u_next):
                raise RuntimeError(f"FA next Conv2d block '{next_name}' expects tensor cache entries.")
            if x_next.dim() != 4 or u_next.dim() != 4:
                raise RuntimeError(f"FA next Conv2d block '{next_name}' expects 4D input/output tensors.")
            if not isinstance(current_layer, nn.Conv2d):
                raise NotImplementedError(
                    f"FA v1 does not support Conv2d->Linear back-mapping on '{current_name}' -> '{next_name}'."
                )

            if delta_next.dim() == 2:
                delta_next = self._lift_linear_feedback_to_conv(delta_next, u_next)
            if delta_next.dim() != 4:
                raise RuntimeError(
                    f"FA expected 4D delta for next Conv2d block '{next_name}', got {tuple(delta_next.shape)}."
                )

            delta_in = torch.nn.grad.conv2d_input(
                x_next.shape,
                feedback,
                delta_next,
                stride=next_layer.stride,
                padding=next_layer.padding,
                dilation=next_layer.dilation,
                groups=next_layer.groups,
            )
            delta_in = self._align_spatial(
                delta_in,
                height=int(u_current.size(-2)),
                width=int(u_current.size(-1)),
            )
            act_grad = activation_derivative_from_preact(self.activation_name, u_current)
            delta_current = delta_in * act_grad

            if x_current.dim() != 4 or u_current.dim() != 4:
                raise RuntimeError(f"FA hidden Conv2d block '{current_name}' expects 4D tensors.")
            assign_conv2d_grads_from_activations_(
                current_layer,
                x_current,
                delta_current,
                average_batch=self.average_grads,
            )
            delta_next = delta_current

        self.step(model.parameters(), require_grads=True, check_finite_grads=True)
        stats = best_effort_stats(objective, out, y)
        self._mark_step_done()
        return stats
