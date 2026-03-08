"""
example_loss.py

Custom loss template for capsule plugins.

Loss plugins also use a 2-step contract:

1. A builder registered with `@register_loss(...)`
2. The builder returns the callable used during training

In other words:

    build_example_scaled_l1(ctx) -> loss_fn(pred, target)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lelabo.losses.registry import LossContext, register_loss


@register_loss("example_scaled_l1")
def build_example_scaled_l1(ctx: LossContext):
    """
    Minimal loss that applies L1 loss and scales the result.

    The builder receives a `LossContext`, which gives access to:
    - `ctx.mode`
    - `ctx.dataset`
    - `ctx.task`
    - `ctx.num_classes`
    - `ctx.loss_params()`

    Example config:
      [loss]
      name = "example_scaled_l1"
      [loss.params]
      scale = 0.5
    """
    params = ctx.loss_params()
    scale = float(params.pop("scale", 1.0))
    module = nn.L1Loss(**params)

    def _loss(pred: torch.Tensor, target: Any) -> torch.Tensor:
        """This is the callable LeLabo will use at each train/eval step."""
        if not torch.is_tensor(target):
            raise TypeError(f"example_scaled_l1 expects tensor targets, got {type(target).__name__}.")
        target_tensor = target.to(device=pred.device, dtype=pred.dtype)
        return module(pred, target_tensor) * scale

    setattr(_loss, "module", module)
    return _loss
