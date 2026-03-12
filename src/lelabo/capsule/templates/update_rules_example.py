"""Simple update-rule template for this capsule.

This example shows a minimal local rule that uses cache views:
- run `forward_with_standard_cache`
- get `views["output_blocks"]`
- write gradients manually on the output head
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache
from lelabo.update_rules.base import OptimizerUpdateRule
from lelabo.update_rules.helpers import assign_linear_grads_from_activations_
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule


def _rule_params(ctx: UpdateRuleContext) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if isinstance(ctx.extra, Mapping):
        extra_params = ctx.extra.get("update_rule_params", {})
        if isinstance(extra_params, Mapping):
            params.update(dict(extra_params))
    from_args = getattr(ctx.args, "update_rule_params", None)
    if isinstance(from_args, Mapping):
        params.update(dict(from_args))
    return params


def _output_delta_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 1:
        probs = torch.softmax(logits.detach(), dim=1)
        targets = torch.zeros_like(probs)
        y_long = y.to(device=logits.device, dtype=torch.long).clamp(min=0, max=max(0, probs.size(1) - 1))
        targets.scatter_(1, y_long.view(-1, 1), 1.0)
        return probs - targets
    return logits.detach() - y.to(device=logits.device, dtype=logits.dtype)


class LocalHeadRule(OptimizerUpdateRule):
    """Tiny local rule: updates only the output Linear head from cache."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        average_grads: bool = False,
        grad_clip: float | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, grad_clip=grad_clip)
        self.average_grads = bool(average_grads)
        self.cache_spec = CacheSpec(
            trainable_module_types=(nn.Linear,),
            observed_module_types=(nn.Linear,),
            capture_inputs=True,
            capture_outputs=True,
            require_single_call=True,
            require_single_output_head=True,
        )

    @torch.no_grad()
    def train_step(self, model, objective, batch, device, state=None) -> dict[str, Any]:
        model.train()
        self.zero_grad()
        _ = objective

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        logits, _cache, views = forward_with_standard_cache(
            model,
            x,
            cache_spec=self.cache_spec,
        )

        output_blocks = views.get("output_blocks", [])
        output_block = output_blocks[0] if isinstance(output_blocks, list) and output_blocks else {}
        output_layer = output_block.get("module")
        x_out = output_block.get("x")

        delta_logits = _output_delta_logits(logits, y)
        if (
            isinstance(output_layer, nn.Linear)
            and torch.is_tensor(x_out)
            and x_out.dim() == 2
            and delta_logits.dim() == 2
            and int(delta_logits.size(1)) == int(output_layer.out_features)
        ):
            assign_linear_grads_from_activations_(
                output_layer,
                x_out,
                delta_logits,
                average_batch=self.average_grads,
            )
            self.step(model.parameters(), require_grads=True, check_finite_grads=True)
        self._mark_step_done()

        stats: dict[str, Any] = {
            "loss_proxy": float((delta_logits * delta_logits).mean().item()),
        }
        if logits.dim() == 2 and y.dim() == 1:
            preds = logits.argmax(dim=1)
            stats["acc"] = float((preds == y.long()).float().mean().item())
        return stats


# @register_update_rule("local_head")
def build_local_head(ctx: UpdateRuleContext):
    params = _rule_params(ctx)
    grad_clip = params.get("grad_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)
    return LocalHeadRule(
        optimizer=ctx.optimizer,
        average_grads=bool(params.get("average_grads", False)),
        grad_clip=grad_clip,
    )
