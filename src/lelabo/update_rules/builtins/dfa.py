from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ..base import OptimizerUpdateRule
from ..helpers import assign_linear_grads_from_activations_
from ...core.batch import extract_loss_and_stats, to_device


class DirectFeedbackAlignment(OptimizerUpdateRule):
    """Simple Linear-only DFA implementation."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        feedback_scale: float = 1.0,
        grad_clip: float | None = None,
        delta_scale: float = 1.0,
        activation: str = "relu",
        average_grads: bool = False,
    ) -> None:
        super().__init__(optimizer=optimizer, grad_clip=grad_clip)
        self.feedback_scale = float(feedback_scale)
        self.delta_scale = float(delta_scale)
        self.activation = str(activation).lower()
        self.average_grads = bool(average_grads)

        self._feedback: dict[str, torch.Tensor] = {}
        self._feedback_shapes: dict[str, tuple[int, int]] = {}

    def _to_2d(self, t: torch.Tensor) -> torch.Tensor:
        return t if t.dim() == 2 else t.view(t.size(0), -1)

    def _activation_grad(self, pre_activation: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return (pre_activation > 0).to(pre_activation.dtype)
        if self.activation == "tanh":
            t = torch.tanh(pre_activation)
            return 1.0 - t * t
        if self.activation == "sigmoid":
            s = torch.sigmoid(pre_activation)
            return s * (1.0 - s)
        if self.activation == "identity":
            return torch.ones_like(pre_activation)
        raise ValueError(f"Unsupported activation for DFA: '{self.activation}'.")

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

    def _output_deltas(self, task, out: Any, y: Any) -> list[tuple[str, torch.Tensor]]:
        if not hasattr(task, "output_deltas"):
            raise NotImplementedError("DFA requires task.output_deltas(out, y).")
        raw = task.output_deltas(out, y)
        if not isinstance(raw, Mapping):
            raise RuntimeError("task.output_deltas must return a dict[str, Tensor].")

        out_deltas: list[tuple[str, torch.Tensor]] = []
        for key, value in raw.items():
            if not torch.is_tensor(value):
                continue
            d = self._to_2d(value)
            if d.numel() > 0:
                out_deltas.append((str(key), d))
        if not out_deltas:
            raise RuntimeError("DFA: task.output_deltas returned no usable tensors.")
        return out_deltas

    def _match_delta_for_head(
        self,
        *,
        block_name: str,
        block_group: str,
        layer: nn.Linear,
        deltas: list[tuple[str, torch.Tensor]],
        used: set[str],
    ) -> torch.Tensor:
        target_dim = int(layer.out_features)
        group_l = block_group.lower()
        name_l = block_name.lower()
        full_l = f"{group_l}.{name_l}"

        def _score(delta_key: str) -> int:
            key_l = delta_key.lower()
            score = 0
            if key_l in full_l or full_l in key_l:
                score += 4
            if key_l in name_l or name_l in key_l:
                score += 2
            if "actor" in group_l and "logits" in key_l:
                score += 3
            if "critic" in group_l and "value" in key_l:
                score += 3
            if ("q" in name_l or "head" in name_l) and "q_values" in key_l:
                score += 3
            if "logits" in name_l and "logits" in key_l:
                score += 1
            if "value" in name_l and "value" in key_l:
                score += 1
            return score

        best_key: str | None = None
        best_delta: torch.Tensor | None = None
        best_score = -10_000

        for key, delta in deltas:
            if key in used or int(delta.size(1)) != target_dim:
                continue
            score = _score(key)
            if score > best_score:
                best_score = score
                best_key = key
                best_delta = delta

        if best_delta is None or best_key is None:
            raise RuntimeError(
                f"DFA: unable to match output delta for block '{block_name}' "
                f"(group='{block_group}', out_features={target_dim})."
            )

        used.add(best_key)
        return best_delta

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

        if isinstance(batch, Mapping):
            raise NotImplementedError("DFA currently supports tuple batches only: (x, y).")

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)

        if not hasattr(model, "get_blocks"):
            raise NotImplementedError("DFA expects model.get_blocks().")

        out_cache = model(x, return_cache=True)
        if not (isinstance(out_cache, tuple) and len(out_cache) == 2):
            raise RuntimeError("DFA expects model(x, return_cache=True) -> (out, cache).")
        out, cache = out_cache

        if not isinstance(cache, Mapping):
            raise RuntimeError("DFA expects cache to be a mapping.")
        block_inputs = cache.get("block_inputs", None)
        if not isinstance(block_inputs, Mapping):
            raise RuntimeError("DFA expects cache['block_inputs'].")

        runtime_blocks = cache.get("block_specs_runtime", None)
        blocks = runtime_blocks if isinstance(runtime_blocks, list) else model.get_blocks()
        linear_blocks = [b for b in blocks if isinstance(getattr(b, "module", None), nn.Linear)]
        if not linear_blocks:
            raise RuntimeError("DFA found no Linear blocks.")

        deltas = self._output_deltas(task, out, y)
        used_delta_keys: set[str] = set()

        blocks_by_group: dict[str, list[Any]] = {}
        for block in linear_blocks:
            group = str(getattr(block, "group", "main") or "main")
            blocks_by_group.setdefault(group, []).append(block)

        for group_name, group_blocks in blocks_by_group.items():
            hidden_blocks = [b for b in group_blocks if not bool(getattr(b, "is_output", False))]
            output_blocks = [b for b in group_blocks if bool(getattr(b, "is_output", False))]
            if not output_blocks:
                continue

            head_deltas: dict[str, torch.Tensor] = {}
            for block in output_blocks:
                name = str(getattr(block, "name", ""))
                layer: nn.Linear = block.module  # type: ignore[assignment]
                head_deltas[name] = self._match_delta_for_head(
                    block_name=name,
                    block_group=group_name,
                    layer=layer,
                    deltas=deltas,
                    used=used_delta_keys,
                )

            delta_out_cat = torch.cat(
                [head_deltas[str(getattr(block, "name", ""))] for block in output_blocks],
                dim=1,
            )
            if self.delta_scale != 1.0:
                delta_out_cat = delta_out_cat * self.delta_scale

            for block in hidden_blocks:
                name = str(getattr(block, "name", ""))
                layer: nn.Linear = block.module  # type: ignore[assignment]
                if name not in block_inputs:
                    raise RuntimeError(f"DFA missing cache['block_inputs'][{name!r}] for hidden block.")
                x_hidden = block_inputs[name]
                if not torch.is_tensor(x_hidden) or x_hidden.dim() != 2:
                    raise RuntimeError(f"DFA hidden block '{name}' expects a 2D tensor input.")

                pre_activation = layer(x_hidden)
                feedback = self._ensure_feedback(
                    key=f"{group_name}:{name}",
                    out_dim=int(delta_out_cat.size(1)),
                    hidden_dim=int(layer.out_features),
                    device=delta_out_cat.device,
                    dtype=delta_out_cat.dtype,
                )
                delta_hidden = (delta_out_cat @ feedback) * self._activation_grad(pre_activation)
                assign_linear_grads_from_activations_(
                    layer,
                    x_hidden,
                    delta_hidden,
                    average_batch=self.average_grads,
                )

            for block in output_blocks:
                name = str(getattr(block, "name", ""))
                layer: nn.Linear = block.module  # type: ignore[assignment]
                if name not in block_inputs:
                    raise RuntimeError(f"DFA missing cache['block_inputs'][{name!r}] for output block.")
                x_out = block_inputs[name]
                if not torch.is_tensor(x_out) or x_out.dim() != 2:
                    raise RuntimeError(f"DFA output block '{name}' expects a 2D tensor input.")

                assign_linear_grads_from_activations_(
                    layer,
                    x_out,
                    head_deltas[name] * self.delta_scale,
                    average_batch=self.average_grads,
                )

        self.step(model.parameters())
        stats = self._best_effort_stats(task, out, y)
        self._mark_step_done()
        return stats

