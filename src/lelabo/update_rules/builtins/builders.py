from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..registry import UpdateRuleContext, register_update_rule
from .backprop import Backpropagation
from .dfa import DirectFeedbackAlignment
from .dni import DNI
from .drtp import DirectRandomTargetProjection
from .fa import FeedbackAlignment
from .scl import SoftContrastiveLearning
from .softhebb import SoftHebb

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
    allowed_keys = {"feedback_scale", "delta_scale", "average_grads", "activation_name", "activation"}
    unknown = sorted(k for k in params.keys() if k not in allowed_keys)
    if unknown:
        raise ValueError(
            f"Unsupported dfa update_rule.params keys: {unknown}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )
    grad_clip = _extra(ctx, "grad_clip", _MISSING)
    if grad_clip is _MISSING:
        grad_clip = _default_grad_clip(ctx)

    activation_name = params.get("activation_name", params.get("activation", None))
    if activation_name is not None:
        activation_name = str(activation_name)

    return DirectFeedbackAlignment(
        optimizer=ctx.optimizer,
        feedback_scale=float(params.get("feedback_scale", 1.0)),
        grad_clip=grad_clip,
        delta_scale=float(params.get("delta_scale", 1.0)),
        average_grads=bool(params.get("average_grads", False)),
        activation_name=activation_name,
    )


@register_update_rule("fa")
def build_fa(ctx: UpdateRuleContext):
    params = _rule_params(ctx)
    allowed_keys = {
        "feedback_scale",
        "delta_scale",
        "average_grads",
        "activation_name",
        "activation",
        "pool_backscale",
    }
    unknown = sorted(k for k in params.keys() if k not in allowed_keys)
    if unknown:
        raise ValueError(
            f"Unsupported fa update_rule.params keys: {unknown}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )
    grad_clip = _extra(ctx, "grad_clip", _MISSING)
    if grad_clip is _MISSING:
        grad_clip = _default_grad_clip(ctx)

    activation_name = params.get("activation_name", params.get("activation", None))
    if activation_name is not None:
        activation_name = str(activation_name)

    return FeedbackAlignment(
        optimizer=ctx.optimizer,
        feedback_scale=float(params.get("feedback_scale", 1.0)),
        grad_clip=grad_clip,
        delta_scale=float(params.get("delta_scale", 1.0)),
        average_grads=bool(params.get("average_grads", False)),
        activation_name=activation_name,
        pool_backscale=bool(params.get("pool_backscale", True)),
    )


@register_update_rule("drtp")
def build_drtp(ctx: UpdateRuleContext):
    params = _rule_params(ctx)
    allowed_keys = {
        "feedback_scale",
        "target_scale",
        "delta_scale",
        "average_grads",
        "activation_name",
        "activation",
    }
    unknown = sorted(k for k in params.keys() if k not in allowed_keys)
    if unknown:
        raise ValueError(
            f"Unsupported drtp update_rule.params keys: {unknown}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )
    grad_clip = _extra(ctx, "grad_clip", _MISSING)
    if grad_clip is _MISSING:
        grad_clip = _default_grad_clip(ctx)

    activation_name = params.get("activation_name", params.get("activation", None))
    if activation_name is not None:
        activation_name = str(activation_name)

    return DirectRandomTargetProjection(
        optimizer=ctx.optimizer,
        feedback_scale=float(params.get("feedback_scale", 1.0)),
        target_scale=float(params.get("target_scale", 1.0)),
        grad_clip=grad_clip,
        delta_scale=float(params.get("delta_scale", 1.0)),
        average_grads=bool(params.get("average_grads", False)),
        activation_name=activation_name,
    )


@register_update_rule("dni")
def build_dni(ctx: UpdateRuleContext):
    params = _rule_params(ctx)
    allowed_keys = {
        "sg_lr",
        "sg_optim",
        "sg_weight_decay",
        "sg_hidden",
        "condition_on_label",
        "lambda_mix",
        "sg_scale",
        "activation",
    }
    unknown = sorted(k for k in params.keys() if k not in allowed_keys)
    if unknown:
        raise ValueError(
            f"Unsupported dni update_rule.params keys: {unknown}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )
    grad_clip = _extra(ctx, "grad_clip", _MISSING)
    if grad_clip is _MISSING:
        grad_clip = _default_grad_clip(ctx)

    return DNI(
        optimizer=ctx.optimizer,
        sg_lr=float(params.get("sg_lr", 3e-4)),
        sg_optim=str(params.get("sg_optim", "adamw")),
        sg_weight_decay=float(params.get("sg_weight_decay", 0.0)),
        sg_hidden=int(params.get("sg_hidden", 0)),
        condition_on_label=bool(params.get("condition_on_label", False)),
        lambda_mix=float(params.get("lambda_mix", 0.0)),
        sg_scale=float(params.get("sg_scale", 1.0)),
        activation=str(params.get("activation", "relu")),
        grad_clip=grad_clip,
    )


@register_update_rule("scl")
def build_scl(ctx: UpdateRuleContext):
    params = _rule_params(ctx)
    allowed_keys = {
        "supcon_tau",
        "local_lr",
        "local_optim",
        "local_weight_decay",
        "proj_dim",
        "proj_hidden_dim",
        "depth_lr_gamma",
        "depth_lr_min_factor",
        "depth_lr_max_factor",
    }
    unknown = sorted(k for k in params.keys() if k not in allowed_keys)
    if unknown:
        raise ValueError(
            f"Unsupported scl update_rule.params keys: {unknown}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )
    grad_clip = _extra(ctx, "grad_clip", _MISSING)
    if grad_clip is _MISSING:
        grad_clip = _default_grad_clip(ctx)

    return SoftContrastiveLearning(
        optimizer=ctx.optimizer,
        supcon_tau=float(params.get("supcon_tau", 0.1)),
        local_lr=(
            None
            if params.get("local_lr", None) is None
            else float(params["local_lr"])
        ),
        local_optim=(
            None
            if params.get("local_optim", None) is None
            else str(params["local_optim"])
        ),
        local_weight_decay=(
            None
            if params.get("local_weight_decay", None) is None
            else float(params["local_weight_decay"])
        ),
        proj_dim=int(params.get("proj_dim", 256)),
        proj_hidden_dim=(
            None if params.get("proj_hidden_dim", None) is None else int(params["proj_hidden_dim"])
        ),
        depth_lr_gamma=float(params.get("depth_lr_gamma", 0.8)),
        depth_lr_min_factor=float(params.get("depth_lr_min_factor", 0.01)),
        depth_lr_max_factor=float(params.get("depth_lr_max_factor", 100.0)),
        grad_clip=grad_clip,
    )


@register_update_rule("softhebb")
def build_softhebb(ctx: UpdateRuleContext):
    params = _rule_params(ctx)
    allowed_keys = {
        "base_lr",
        "lr_conv1",
        "lr_conv2",
        "lr_conv3",
        "power_lr",
        "unsup_epochs",
        "sup_epochs",
        "steps_per_epoch",
        "eps_norm",
        "conv_t_invert",
    }
    unknown_keys = sorted(k for k in params if k not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unsupported softhebb update_rule.params keys: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    grad_clip = _extra(ctx, "grad_clip", _MISSING)
    if grad_clip is _MISSING:
        grad_clip = _default_grad_clip(ctx)

    kwargs: dict[str, Any] = {
        "optimizer": ctx.optimizer,
        "head_optimizer": ctx.optimizer,
        "grad_clip": grad_clip,
    }
    for key in sorted(allowed_keys):
        if key in params:
            kwargs[key] = params[key]
    return SoftHebb(**kwargs)
