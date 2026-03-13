"""
example.py

Purpose
-------
Add a simple model plugin to this capsule.

Contract
--------
1. Register a builder with `@register_model("example_mlp")`
2. The builder receives `(ctx, args)`
3. The builder returns an `nn.Module`

Where params come from
----------------------
Read model-specific params from `args.model_params`.
For exact `ModelContext` fields, read `resources/LELABO_REFERENCE.md`.
For the param mapping, read `resources/PARAM_FLOW.md`.

Official example
----------------
`ExampleMLP` is a compact tabular classifier that uses `ctx.in_dim`
and `ctx.num_classes`.

Config snippet
--------------
[model]
name = "example_mlp"

[model.params]
hidden = 128
layers = 3
dropout = 0.1

How to activate
---------------
Uncomment `@register_model("example_mlp")`.

How to test
-----------
lelabo list models
lelabo train supervised --config configs/train.supervised.quickstart.toml --model example_mlp

Common errors
-------------
- `ctx.in_dim` is missing: this example expects a tabular input dimension.
- Need cache/block-level behavior for a local rule: read
  `resources/MODEL_CACHE_ADVANCED.md` and `models/cache_walkthrough.py`.
"""

from __future__ import annotations

import torch.nn as nn

from lelabo.models.registry import ModelContext, register_model


class ExampleMLP(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        num_classes: int,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden)
        num_layers = int(max(1, layers))
        dropout_p = float(max(0.0, dropout))

        blocks: list[nn.Module] = []
        current_dim = int(in_dim)
        for _ in range(num_layers):
            blocks.append(nn.Linear(current_dim, hidden_dim))
            blocks.append(nn.ReLU())
            if dropout_p > 0.0:
                blocks.append(nn.Dropout(dropout_p))
            current_dim = hidden_dim
        blocks.append(nn.Linear(current_dim, int(num_classes)))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# @register_model("example_mlp")
def build_example_mlp(ctx: ModelContext, args) -> ExampleMLP:
    params = dict(getattr(args, "model_params", {}) or {})
    if ctx.in_dim is None:
        raise ValueError("example_mlp needs `ctx.in_dim`. Use it with a tabular dataset or provide input_shape.")
    return ExampleMLP(
        in_dim=int(ctx.in_dim),
        num_classes=int(ctx.num_classes),
        hidden=int(params.get("hidden", 128)),
        layers=int(params.get("layers", 3)),
        dropout=float(params.get("dropout", 0.0)),
    )
