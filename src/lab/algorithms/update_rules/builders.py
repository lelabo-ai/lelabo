from __future__ import annotations

from typing import Any

import torch

from .backprop import Backprop
from .dfa import DirectFeedbackAlignment
from .dni import DNI
from .feedbackalignment import FeedbackAlignment
from .kp import KP
from .kp3 import KP3
from .local_probe_bert import LocalProbeBERT
from .local_probe_blocks import LocalProbeBlocks
from .local_probe_mlp import LocalProbeMLP
from .scl import SoftContrastiveLearning
from .softhebb import SoftHebb
from .targetprop import TargetPropagation
from .registry import UpdateRuleContext, register_update_rule
from .kp2 import KP2

_MISSING = object()


def _extra(ctx: UpdateRuleContext, key: str, default: Any = _MISSING):
    if ctx.extra is None:
        return default
    if key in ctx.extra:
        return ctx.extra[key]
    return default


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


@register_update_rule("lpl")
def build_local_probe(ctx: UpdateRuleContext):
    if ctx.mode != "supervised":
        raise ValueError("lpl is only supported in supervised mode.")
    if ctx.dataset == "glue":
        return LocalProbeBERT(base_optimizer=ctx.optimizer, probe_lr=ctx.args.lr)
    if getattr(ctx.args, "model", None) == "mlp":
        return LocalProbeMLP(base_optimizer=ctx.optimizer, probe_lr=ctx.args.lr, weight_decay=ctx.args.weight_decay)
    return LocalProbeBlocks(base_optimizer=ctx.optimizer, probe_lr=ctx.args.lr)


@register_update_rule("kp")
def build_kp(ctx: UpdateRuleContext):
    return KP(learning_rate=ctx.args.lr, bp_lr=ctx.args.lr, bp_weight_decay=ctx.args.weight_decay)


@register_update_rule("scl")
def build_scl(ctx: UpdateRuleContext):
    return SoftContrastiveLearning(local_lr=ctx.args.lr, head_lr=ctx.args.lr, local_weight_decay=ctx.args.weight_decay)

@register_update_rule("kp2")
def build_kp2(ctx: UpdateRuleContext):
    return KP2(learning_rate=ctx.args.lr, head_lr=ctx.args.lr, head_weight_decay=ctx.args.weight_decay)

@register_update_rule("kp3")
def build_kp3(ctx: UpdateRuleContext):
    return KP3(local_lr=ctx.args.lr, head_lr=ctx.args.lr, head_weight_decay=ctx.args.weight_decay)


@register_update_rule("softhebb")
def build_softhebb(ctx: UpdateRuleContext):
    return SoftHebb(head_lr=ctx.args.lr)


@register_update_rule("tp")
def build_targetprop(ctx: UpdateRuleContext):
    if ctx.mode == "supervised":
        dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=True)
        inv_optim = torch.optim.SGD([dummy], lr=ctx.args.lr)
        return TargetPropagation(
            fwd_optimizer=ctx.optimizer,
            inv_optimizer=inv_optim,
            beta=1.0,
            noise_std=0.1,
        )
    return TargetPropagation(
        fwd_lr=ctx.args.lr,
        inv_lr=ctx.args.lr,
        fwd_optimizer=ctx.optimizer,
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
