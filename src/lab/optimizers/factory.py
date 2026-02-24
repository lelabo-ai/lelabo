# lab/optimizers/factory.py
from __future__ import annotations

from typing import Iterable, Any, Optional, Sequence
import torch


def make_optimizer(
    name: str,
    params: Iterable,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    **kwargs: Any,
):
    name = name.lower()

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)

    if name in ["sgd+momentum", "sgd_momentum", "momentum"]:
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            **kwargs,
        )

    if name == "ano":
        from ano_optimizer import Ano  # type: ignore
        return Ano(params, lr=lr, weight_decay=weight_decay, **kwargs)

    raise ValueError(f"Unknown optimizer: {name}")


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


def make_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    interval: str = "epoch",
    monitor: str = "val.loss",
    **kwargs: Any,
):
    """
    Build a torch LR scheduler from a short name + kwargs.
    Returns None if name is None/"none".
    """
    if name is None:
        return None
    name = str(name).lower()
    if name in ("", "none", "null", "off"):
        return None

    interval = str(interval).lower()

    if name in ("step", "steplr"):
        step_size = int(kwargs.pop("step_size", 10))
        gamma = float(kwargs.pop("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)

    if name in ("multistep", "multi_step", "multisteplr"):
        milestones = _parse_int_list(kwargs.pop("milestones", None))
        if not milestones:
            milestones = [30, 60]
        gamma = float(kwargs.pop("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, **kwargs)

    if name in ("exponential", "explr"):
        gamma = float(kwargs.pop("gamma", 0.99))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, **kwargs)

    if name in ("cosine", "cosineannealing", "cosineannealinglr"):
        t_max = kwargs.pop("T_max", None)
        if t_max is None:
            if interval == "batch" and epochs is not None and steps_per_epoch is not None:
                t_max = int(epochs * steps_per_epoch)
            elif epochs is not None:
                t_max = int(epochs)
            else:
                t_max = 10
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(t_max), **kwargs)

    if name in ("cosinewarm", "cosinewarmrestarts", "cosineannealingwarmrestarts"):
        t0 = int(kwargs.pop("T_0", 10))
        tmult = int(kwargs.pop("T_mult", 1))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=tmult, **kwargs)

    if name in ("plateau", "reducelronplateau", "reduceonplateau"):
        mode = kwargs.pop("mode", None)
        if mode is None:
            mode = "max" if ("acc" in monitor or "metric" in monitor) else "min"
        factor = float(kwargs.pop("factor", 0.1))
        patience = int(kwargs.pop("patience", 10))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            **kwargs,
        )

    if name in ("onecycle", "onecyclelr"):
        max_lr = kwargs.pop("max_lr", None)
        if max_lr is None:
            raise ValueError("OneCycleLR requires max_lr (pass via --lr-scheduler-kwargs).")

        total_steps = kwargs.pop("total_steps", None)
        if total_steps is None:
            if epochs is not None and steps_per_epoch is not None:
                total_steps = int(epochs * steps_per_epoch)
            else:
                raise ValueError("OneCycleLR requires total_steps or (epochs + steps_per_epoch).")
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=int(total_steps), **kwargs)

    if name in ("linear", "linearlr"):
        total_iters = kwargs.pop("total_iters", None)
        if total_iters is None:
            if interval == "batch" and epochs is not None and steps_per_epoch is not None:
                total_iters = int(epochs * steps_per_epoch)
            elif epochs is not None:
                total_iters = int(epochs)
            else:
                total_iters = 10
        return torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=int(total_iters), **kwargs)

    if name in ("constant", "constantlr"):
        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)

    raise ValueError(f"Unknown scheduler: {name}")
