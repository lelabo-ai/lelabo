"""
example.py

Purpose
-------
Add a custom optimizer builder to this capsule.

Contract
--------
1. Register a builder with `@register_optimizer("capsule_sgd")`
2. The builder receives an `OptimizerContext`
3. The builder returns a `torch.optim.Optimizer`

Where params come from
----------------------
For the full add-an-optimizer procedure, read `resources/EXTENSION_RECIPES.md`.
Use `ctx.lr`, `ctx.weight_decay`, and `ctx.momentum` for normalized fields.
Use `ctx.optimizer_params()` for extra optimizer params.
For exact context fields, read `resources/LELABO_REFERENCE.md`.
For the param mapping, read `resources/PARAM_FLOW.md`.

Official example
----------------
`capsule_sgd` is the official first capsule path:
`mnist + cnn + bp + capsule_sgd`

Config snippet
--------------
[optimizer]
name = "capsule_sgd"

[optimizer.params]
lr = 0.05
weight_decay = 0.0005

How to activate
---------------
Uncomment `@register_optimizer("capsule_sgd")`.

How to test
-----------
lelabo list optimizers
lelabo train supervised --config configs/train/supervised.capsule_optimizer.toml
pytest -q tests

Common errors
-------------
- The optimizer does not appear in `lelabo list optimizers`: the decorator is still commented.
- The builder must return a real `torch.optim.Optimizer`.
"""

from __future__ import annotations

import torch

from lelabo.optimizers import OptimizerContext, register_optimizer


# @register_optimizer("capsule_sgd")
def build_capsule_sgd(ctx: OptimizerContext):
    return torch.optim.SGD(
        ctx.params,
        lr=ctx.lr,
        weight_decay=ctx.weight_decay,
    )
