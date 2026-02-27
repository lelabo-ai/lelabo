from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import UpdateRuleContext, register_update_rule
from .backprop import Backpropagation
from .dfa import DirectFeedbackAlignment

_MISSING = object()


def _extra(ctx: UpdateRuleContext, key: str, default: Any = _MISSING):
    if ctx.extra is None:
        return default
    if key in ctx.extra:
        return ctx.extra[key]
    return default


def _rule_params(ctx: UpdateRuleContext) -> dict[str, Any]:
    params: dict[str, Any] = {}
    from_extra = _extra(ctx, "update_rule_params", {})
    if isinstance(from_extra, Mapping):
        params.update(dict(from_extra))
    from_args = getattr(ctx.args, "update_rule_params", None)
    if isinstance(from_args, Mapping):
        params.update(dict(from_args))
    return params


def _default_grad_clip(ctx: UpdateRuleContext) -> float | None:
    if ctx.mode == "rl" and ctx.rl_algo == "ppo":
        return getattr(ctx.args, "max_grad_norm", None)
    return None


@register_update_rule("bp")
def build_backprop(ctx: UpdateRuleContext):
    grad_clip = _extra(ctx, "grad_clip", _MISSING)
    if grad_clip is _MISSING:
        grad_clip = _default_grad_clip(ctx)
    return Backpropagation(optimizer=ctx.optimizer, grad_clip=grad_clip)


@register_update_rule("dfa")
def build_dfa(ctx: UpdateRuleContext):
    params = _rule_params(ctx)
    grad_clip = _extra(ctx, "grad_clip", _MISSING)
    if grad_clip is _MISSING:
        grad_clip = _default_grad_clip(ctx)

    return DirectFeedbackAlignment(
        optimizer=ctx.optimizer,
        feedback_scale=float(params.get("feedback_scale", 1.0)),
        grad_clip=grad_clip,
        delta_scale=float(params.get("delta_scale", 1.0)),
        activation=str(params.get("activation", "relu")),
        average_grads=bool(params.get("average_grads", False)),
    )

