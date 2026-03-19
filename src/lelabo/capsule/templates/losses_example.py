"""
example.py

Purpose
-------
Add a custom loss to this capsule.

Contract
--------
1. Register a builder with `@register_loss("example_scaled_l1")`
2. The builder receives a `LossContext`
3. The builder returns the callable used during training

Where params come from
----------------------
For the extension recipe, read `resources/EXTENSION_RECIPES.md`.
Read params with `ctx.loss_params()`.
For exact context fields, read `resources/LELABO_REFERENCE.md`.
For the param mapping, read `resources/PARAM_FLOW.md`.

Official example
----------------
`example_scaled_l1` wraps `nn.L1Loss` and multiplies the result by `scale`.

Config snippet
--------------
[loss]
name = "example_scaled_l1"

[loss.params]
scale = 0.5

How to activate
---------------
Uncomment `@register_loss("example_scaled_l1")`.

How to test
-----------
lelabo list losses
lelabo train supervised --config configs/train/supervised.quickstart.toml --loss example_scaled_l1

Common errors
-------------
- The returned callable must accept `(pred, target)`.
- Convert targets to the right device/dtype before applying the loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lelabo.losses.registry import LossContext, register_loss


# @register_loss("example_scaled_l1")
def build_example_scaled_l1(ctx: LossContext):
    params = ctx.loss_params()
    scale = float(params.pop("scale", 1.0))
    module = nn.L1Loss(**params)

    def _loss(pred: torch.Tensor, target: Any) -> torch.Tensor:
        if not torch.is_tensor(target):
            raise TypeError(f"example_scaled_l1 expects tensor targets, got {type(target).__name__}.")
        target_tensor = target.to(device=pred.device, dtype=pred.dtype)
        return module(pred, target_tensor) * scale

    setattr(_loss, "module", module)
    return _loss
