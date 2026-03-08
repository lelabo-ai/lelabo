"""
example_initializer.py

Example custom initializer plugin for LeLabo.

This example uses a small helper so the plugin itself stays focused on the
custom initialization logic.

Initializer plugins follow a 2-step contract:

1. Register a builder with `@register_initializer(...)`
2. Return a callable that receives the instantiated model and initializes it in-place

In other words:

    build_row_sum_one(ctx) -> apply_initializer(model)
"""

from __future__ import annotations

import torch

from lelabo.initializers.registry import InitializerContext, register_initializer

from .initializer_helpers import make_initializer


def _normalize_output_sums_(w: torch.Tensor, eps: float) -> None:
    """
    Normalize each output unit/channel so its incoming weights sum to 1.

    - Linear: [out_features, in_features]
    - ConvNd: [out_channels, in_channels, ...]
    """
    if w.dim() < 2:
        return

    flat = w.view(w.shape[0], -1)
    sums = flat.sum(dim=1, keepdim=True).clamp_min(eps)
    flat.div_(sums)


#@register_initializer("row_sum_one")
def build_row_sum_one(ctx: InitializerContext):
    """
    Initialize Linear/Conv weights so that each output neuron/channel receives
    weights whose sum is equal to 1.

    Specific parameters in `[initializer.params]`:
      - distribution: "uniform" | "normal", default = "uniform"
      - mean: float, used when distribution = "normal", default = 0.0
      - std: float, used when distribution = "normal", default = 0.02
      - eps: float, numerical stability term, default = 1e-12

    Generic parameters also supported through the helper:
      - bias: "zeros" | "none"
      - norm_weight: "ones" | "zeros" | "none"
      - norm_bias: "ones" | "zeros" | "none"
      - seed: optional integer

    Example config:

      [initializer]
      name = "row_sum_one"

      [initializer.params]
      distribution = "uniform"
      bias = "zeros"
      norm_weight = "ones"
      norm_bias = "zeros"
      seed = 1234
    """
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
