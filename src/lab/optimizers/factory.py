# lab/optimizers/factory.py
from __future__ import annotations

from typing import Iterable, Any, Optional
import torch

from .registry import make_optimizer as _make_optimizer
from ..schedulers import make_scheduler as _make_scheduler


def make_optimizer(
    name: str,
    params: Iterable,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    *,
    args: Any | None = None,
    mode: str | None = None,
    dataset: str | None = None,
    extra: dict[str, Any] | None = None,
    **kwargs: Any,
):
    return _make_optimizer(
        name=name,
        params=params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        args=args,
        mode=mode,
        dataset=dataset,
        extra=extra,
        **kwargs,
    )


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
    return _make_scheduler(
        name=name,
        optimizer=optimizer,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        interval=interval,
        monitor=monitor,
        **kwargs,
    )
