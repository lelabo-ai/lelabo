"""
example.py

Purpose
-------
Add a scheduler builder to this capsule.

Contract
--------
1. Register a builder with `@register_scheduler("my_cosine")`
2. The builder receives a `SchedulerContext`
3. The builder returns the scheduler/controller used by LeLabo

Where params come from
----------------------
For the extension recipe, read `resources/EXTENSION_RECIPES.md`.
Use `ctx.interval`, `ctx.monitor`, `ctx.epochs`, and `ctx.steps_per_epoch`
for normalized scheduling metadata.
Use `ctx.scheduler_params()` for extra scheduler params.
For exact context fields, read `resources/LELABO_REFERENCE.md`.
For the param mapping, read `resources/PARAM_FLOW.md`.

Official example
----------------
`my_cosine` wraps `torch.optim.lr_scheduler.CosineAnnealingLR`.

Config snippet
--------------
[scheduler]
name = "my_cosine"
interval = "epoch"
monitor = "val.loss"

[scheduler.params]
T_max = 50
eta_min = 0.0

How to activate
---------------
Uncomment `@register_scheduler("my_cosine")`.

How to test
-----------
lelabo registries schedulers
lelabo train supervised --config configs/train/supervised.detailed.toml --scheduler my_cosine

Common errors
-------------
- If `T_max` is missing and the training horizon is unknown, this example raises a clear error.
- The builder must return a scheduler compatible with the LeLabo scheduler controller.
"""

from __future__ import annotations

import torch

from lelabo.schedulers import SchedulerContext, register_scheduler


# @register_scheduler("my_cosine")
def build_my_cosine(ctx: SchedulerContext):
    params = ctx.scheduler_params()
    t_max = params.pop("T_max", None)
    if t_max is None:
        if ctx.epochs is None:
            raise ValueError("my_cosine requires T_max or a known epochs horizon.")
        t_max = int(ctx.epochs)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        ctx.optimizer,
        T_max=int(t_max),
        **params,
    )
