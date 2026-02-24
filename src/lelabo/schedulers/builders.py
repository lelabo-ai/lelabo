from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .registry import SchedulerContext, register_scheduler


def _parse_int_list(val: Any) -> list[int]:
    if val is None:
        return []
    if isinstance(val, str):
        if val.strip() == "":
            return []
        return [int(x) for x in val.split(",") if x.strip()]
    if isinstance(val, Sequence):
        return [int(x) for x in val]
    return [int(val)]


def _resolve_epoch_or_batch_horizon(ctx: SchedulerContext, *, fallback: int = 10) -> int:
    interval = str(ctx.interval).strip().lower()
    if interval in {"batch", "step"}:
        if ctx.total_steps is not None:
            return int(ctx.total_steps)
    if ctx.epochs is not None:
        try:
            epochs = int(ctx.epochs)
            if epochs > 0:
                return epochs
        except Exception:
            pass
    return int(fallback)


@register_scheduler("steplr")
def build_step_lr(ctx: SchedulerContext):
    kwargs = ctx.scheduler_params()
    step_size = int(kwargs.pop("step_size", 10))
    gamma = float(kwargs.pop("gamma", 0.1))
    return torch.optim.lr_scheduler.StepLR(ctx.optimizer, step_size=step_size, gamma=gamma, **kwargs)


@register_scheduler("multisteplr")
def build_multistep_lr(ctx: SchedulerContext):
    kwargs = ctx.scheduler_params()
    milestones = _parse_int_list(kwargs.pop("milestones", None))
    if not milestones:
        milestones = [30, 60]
    gamma = float(kwargs.pop("gamma", 0.1))
    return torch.optim.lr_scheduler.MultiStepLR(ctx.optimizer, milestones=milestones, gamma=gamma, **kwargs)


@register_scheduler("exponentiallr")
def build_exponential_lr(ctx: SchedulerContext):
    kwargs = ctx.scheduler_params()
    gamma = float(kwargs.pop("gamma", 0.99))
    return torch.optim.lr_scheduler.ExponentialLR(ctx.optimizer, gamma=gamma, **kwargs)


@register_scheduler("cosineannealinglr")
def build_cosine_annealing_lr(ctx: SchedulerContext):
    kwargs = ctx.scheduler_params()
    t_max = kwargs.pop("T_max", None)
    if t_max is None:
        t_max = _resolve_epoch_or_batch_horizon(ctx, fallback=10)
    return torch.optim.lr_scheduler.CosineAnnealingLR(ctx.optimizer, T_max=int(t_max), **kwargs)


@register_scheduler("cosineannealingwarmrestarts")
def build_cosine_warm_restarts(ctx: SchedulerContext):
    kwargs = ctx.scheduler_params()
    t0 = int(kwargs.pop("T_0", 10))
    tmult = int(kwargs.pop("T_mult", 1))
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(ctx.optimizer, T_0=t0, T_mult=tmult, **kwargs)


@register_scheduler("reducelronplateau")
def build_reduce_on_plateau(ctx: SchedulerContext):
    kwargs = ctx.scheduler_params()
    mode = kwargs.pop("mode", None)
    if mode is None:
        monitor = str(ctx.monitor).strip().lower()
        mode = "max" if ("acc" in monitor or "metric" in monitor) else "min"
    factor = float(kwargs.pop("factor", 0.1))
    patience = int(kwargs.pop("patience", 10))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        ctx.optimizer,
        mode=str(mode),
        factor=factor,
        patience=patience,
        **kwargs,
    )


@register_scheduler("onecyclelr")
def build_onecycle_lr(ctx: SchedulerContext):
    interval = str(ctx.interval).strip().lower()
    if interval not in {"batch", "step"}:
        raise ValueError("OneCycleLR requires scheduler.interval='batch'.")

    kwargs = ctx.scheduler_params()
    max_lr = kwargs.pop("max_lr", None)
    if max_lr is None:
        raise ValueError("OneCycleLR requires max_lr.")

    total_steps = kwargs.pop("total_steps", None)
    if total_steps is None:
        total_steps = ctx.total_steps
    if total_steps is None:
        raise ValueError("OneCycleLR requires total_steps or (epochs + steps_per_epoch).")
    return torch.optim.lr_scheduler.OneCycleLR(
        ctx.optimizer,
        max_lr=max_lr,
        total_steps=int(total_steps),
        **kwargs,
    )


@register_scheduler("linearlr")
def build_linear_lr(ctx: SchedulerContext):
    kwargs = ctx.scheduler_params()
    total_iters = kwargs.pop("total_iters", None)
    if total_iters is None:
        total_iters = _resolve_epoch_or_batch_horizon(ctx, fallback=10)
    return torch.optim.lr_scheduler.LinearLR(ctx.optimizer, total_iters=int(total_iters), **kwargs)


@register_scheduler("constantlr")
def build_constant_lr(ctx: SchedulerContext):
    kwargs = ctx.scheduler_params()
    return torch.optim.lr_scheduler.ConstantLR(ctx.optimizer, **kwargs)
