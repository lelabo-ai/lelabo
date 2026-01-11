# lab/core/optim.py
from __future__ import annotations

from typing import Iterable
import torch


def make_optimizer(
    name: str,
    params: Iterable,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
):
    name = name.lower()

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)

    if name in ["sgd+momentum", "sgd_momentum", "momentum"]:
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)

    if name == "ano":
        from ano_optimizer import Ano  # type: ignore
        return Ano(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {name}")
