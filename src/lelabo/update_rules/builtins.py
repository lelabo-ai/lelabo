from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import torch

from .backprop import Backprop
from .dfa import DirectFeedbackAlignment
from .dni import DNI
from .feedbackalignment import FeedbackAlignment
from .scl import SoftContrastiveLearning
from .softhebb import SoftHebb
from .targetprop import TargetPropagation
from .registry import UpdateRuleContext, register_update_rule

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


def _grad_clip_for_dataset(dataset: str | None) -> float | None:
    if dataset in {"mnist", "glue"}:
        return 1.0
    return None


@register_update_rule("bp")
def build_backprop(ctx: UpdateRuleContext):
    override = _extra(ctx, "grad_clip")
    if override is not _MISSING:
        grad_clip = override
    elif ctx.mode == "supervised":
        grad_clip = _grad_clip_for_dataset(ctx.dataset)
    elif ctx.mode == "rl" and ctx.rl_algo == "ppo":
        grad_clip = getattr(ctx.args, "max_grad_norm", None)
    else:
        grad_clip = None
    return Backprop(optimizer=ctx.optimizer, grad_clip=grad_clip)

@register_update_rule("scl")
def build_scl(ctx: UpdateRuleContext):
    return SoftContrastiveLearning(local_lr=ctx.args.lr, head_lr=ctx.args.lr, local_weight_decay=ctx.args.weight_decay)


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
    kwargs: dict[str, Any] = {
        # Reuse optimizer created in supervised/rl runner.
        "head_optimizer": ctx.optimizer,
    }
    for key in sorted(allowed_keys):
        if key in params:
            kwargs[key] = params[key]
    return SoftHebb(**kwargs)


@register_update_rule("tp")
def build_targetprop(ctx: UpdateRuleContext):
    params = _rule_params(ctx)
    return TargetPropagation(
        fwd_lr=float(params.get("fwd_lr", ctx.args.lr)),
        inv_lr=float(params.get("inv_lr", ctx.args.lr)),
        beta=float(params.get("beta", 1.0)),
        noise_std=float(params.get("noise_std", 0.1)),
        eps=float(params.get("eps", 1e-8)),
        fwd_optimizer=ctx.optimizer,
        # User requirement: share the exact same optimizer object.
        inv_optimizer=ctx.optimizer,
    )


@register_update_rule("fa")
def build_fa(ctx: UpdateRuleContext):
    override = _extra(ctx, "grad_clip")
    if override is not _MISSING:
        grad_clip = override
    elif ctx.mode == "supervised":
        grad_clip = _grad_clip_for_dataset(ctx.dataset)
    else:
        grad_clip = None
    return FeedbackAlignment(optimizer=ctx.optimizer, grad_clip=grad_clip)


@register_update_rule("dfa")
def build_dfa(ctx: UpdateRuleContext):
    override = _extra(ctx, "grad_clip")
    if override is not _MISSING:
        grad_clip = override
    elif ctx.mode == "supervised":
        grad_clip = _grad_clip_for_dataset(ctx.dataset)
    else:
        grad_clip = None
    return DirectFeedbackAlignment(optimizer=ctx.optimizer, grad_clip=grad_clip)


@register_update_rule("dni")
def build_dni(ctx: UpdateRuleContext):
    if ctx.mode == "supervised":
        return DNI(
            lr=ctx.args.lr,
            sg_lr=ctx.args.lr,
            sg_hidden=0,
            condition_on_label=False,
            lambda_mix=0.0,
            sg_scale=1.0,
            activation="relu",
        )
    return DNI(
        lr=ctx.args.lr,
        sg_lr=ctx.args.lr,
        net_weight_decay=ctx.args.weight_decay,
        sg_weight_decay=ctx.args.weight_decay,
    )
