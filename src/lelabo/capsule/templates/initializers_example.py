"""
example.py

Purpose
-------
Add a custom initializer to this capsule.

Contract
--------
1. Register a builder with `@register_initializer("row_sum_one")`
2. The builder receives an `InitializerContext`
3. The builder returns a callable that initializes a model in-place

Where params come from
----------------------
For the extension recipe, read `resources/EXTENSION_RECIPES.md`.
Read params with `ctx.initializer_params()`.
For exact context fields, read `resources/LELABO_REFERENCE.md`.
For the param mapping, read `resources/PARAM_FLOW.md`.

Official example
----------------
`row_sum_one` initializes Linear/Conv weights and then normalizes each output
unit so its incoming weights sum to 1.

Config snippet
--------------
[initializer]
name = "row_sum_one"

[initializer.params]
distribution = "uniform"
bias = "zeros"
norm_weight = "ones"
norm_bias = "zeros"
seed = 1234

How to activate
---------------
Uncomment `@register_initializer("row_sum_one")`.

How to test
-----------
lelabo list initializers
lelabo train supervised --config configs/train/supervised.quickstart.toml --initializer row_sum_one

Common errors
-------------
- The builder must return `callable(model)`.
- Use `initializer_helpers.py` for shared validation and logging behavior.
"""

from __future__ import annotations

import torch

from lelabo.initializers.registry import InitializerContext, register_initializer

from .initializer_helpers import make_initializer


def _normalize_output_sums_(w: torch.Tensor, eps: float) -> None:
    if w.dim() < 2:
        return

    flat = w.view(w.shape[0], -1)
    sums = flat.sum(dim=1, keepdim=True).clamp_min(eps)
    flat.div_(sums)


# @register_initializer("row_sum_one")
def build_row_sum_one(ctx: InitializerContext):
    params = ctx.initializer_params()

    distribution = str(params.get("distribution", "uniform")).strip().lower()
    mean = float(params.get("mean", 0.0))
    std = float(params.get("std", 0.02))
    eps = float(params.get("eps", 1e-12))

    if distribution not in {"uniform", "normal"}:
        raise ValueError("initializer.params.distribution must be one of: uniform, normal.")
    if eps <= 0:
        raise ValueError("initializer.params.eps must be > 0.")
    if distribution == "normal" and std <= 0:
        raise ValueError("initializer.params.std must be > 0 when distribution='normal'.")

    def _weight_init(w: torch.Tensor) -> None:
        if distribution == "uniform":
            w.uniform_(0.0, 1.0)
        else:
            w.normal_(mean=mean, std=std)
        _normalize_output_sums_(w, eps=eps)

    return make_initializer(
        ctx,
        weight_init=_weight_init,
        name="row_sum_one",
        extra_log=lambda: {
            "distribution": distribution,
            "mean": mean,
            "std": std,
            "eps": eps,
        },
    )
