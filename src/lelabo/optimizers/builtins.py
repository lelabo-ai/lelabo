from __future__ import annotations

import torch

from .registry import OptimizerContext, register_optimizer


@register_optimizer("adamw")
def build_adamw(ctx: OptimizerContext):
    kwargs = ctx.optimizer_params()
    return torch.optim.AdamW(
        ctx.params,
        lr=float(ctx.lr),
        weight_decay=float(ctx.weight_decay),
        **kwargs,
    )


@register_optimizer("adam")
def build_adam(ctx: OptimizerContext):
    kwargs = ctx.optimizer_params()
    return torch.optim.Adam(
        ctx.params,
        lr=float(ctx.lr),
        weight_decay=float(ctx.weight_decay),
        **kwargs,
    )


@register_optimizer("sgd")
def build_sgd(ctx: OptimizerContext):
    kwargs = ctx.optimizer_params()
    return torch.optim.SGD(
        ctx.params,
        lr=float(ctx.lr),
        weight_decay=float(ctx.weight_decay),
        **kwargs,
    )


@register_optimizer("sgd+momentum")
def build_sgd_momentum(ctx: OptimizerContext):
    kwargs = ctx.optimizer_params()
    return torch.optim.SGD(
        ctx.params,
        lr=float(ctx.lr),
        weight_decay=float(ctx.weight_decay),
        momentum=float(ctx.momentum),
        **kwargs,
    )


@register_optimizer("rmsprop")
def build_rmsprop(ctx: OptimizerContext):
    kwargs = ctx.optimizer_params()
    return torch.optim.RMSprop(
        ctx.params,
        lr=float(ctx.lr),
        weight_decay=float(ctx.weight_decay),
        momentum=float(kwargs.pop("momentum", 0.0)),
        **kwargs,
    )


@register_optimizer("adagrad")
def build_adagrad(ctx: OptimizerContext):
    kwargs = ctx.optimizer_params()
    return torch.optim.Adagrad(
        ctx.params,
        lr=float(ctx.lr),
        weight_decay=float(ctx.weight_decay),
        **kwargs,
    )


@register_optimizer("ano")
def build_ano(ctx: OptimizerContext):
    kwargs = ctx.optimizer_params()
    try:
        from ano_optimizer import Ano  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Optimizer 'ano' requires optional dependency 'ano_optimizer'. Install it to use this optimizer."
        ) from exc
    return Ano(
        ctx.params,
        lr=float(ctx.lr),
        weight_decay=float(ctx.weight_decay),
        **kwargs,
    )

