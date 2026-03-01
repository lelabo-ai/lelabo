"""
example.py

Minimal documented example showing:

1) How to register a model
2) How to keep model API pure nn.Module

We strongly recommend for the model implementation to NOT use F.xxx functional APIs, as they can make it difficult for the autocache framework 
to identify module boundaries and capture activations. It will so result in more limited cache availability and less compatibility for different
update rules like DFA/TP/SoftHebb. However, if you don't want to use nn.Module for some reason, the backpropagation rules still work perfectly fine.

We also used this example models to demonstrate how the autocache framework can be used to create new local rules. 
See __name__ == "__main__" block for example usage of forward_with_standard_cache with different CacheSpec settings.
"""

from __future__ import annotations

from collections.abc import Mapping
import torch
import torch.nn as nn

from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache
from lelabo.models.registry import ModelContext


class ExampleMLP(nn.Module):
    """Minimal MLP example with pure PyTorch API."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_classes: int,
        *,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.activation_name = str(activation).lower()
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        def _make_act(name: str) -> nn.Module:
            if name == "relu":
                return nn.ReLU()
            if name == "tanh":
                return nn.Tanh()
            if name == "sigmoid":
                return nn.Sigmoid()
            if name in {"identity", "none", "linear"}:
                return nn.Identity()
            raise ValueError(
                f"Unsupported activation '{name}'. "
                "Supported: relu, tanh, sigmoid, identity."
            )

        blocks: list[nn.Module] = []
        in_features = self.in_dim
        for _ in range(self.num_layers):
            blocks.append(nn.Linear(in_features, self.hidden))
            blocks.append(_make_act(self.activation_name))
            in_features = self.hidden
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        return self.head(h)


# @register_model("example_mlp")
def build_example_mlp(ctx: ModelContext, args):
    """
    Builder pattern used by LeLabo models.

    Priority order for model params:
    1) `model.params.<key>` from config (available as `args.model_params`)
    2) flat CLI args (`args.<key>`) for backward compatibility

    Example config:
      [model]
      name = "example_mlp"
      [model.params]
      hidden = 256
      layers = 3
      activation = "tanh"
      
    This function can be simplier if you don't need the flexible param naming or priority, 
    but we recommend following this pattern for better compatibility with different config 
    styles and future extensions.
    
    Moreover, it can also be very useful for ablations study to compare different param 
    settings without changing the config structure (e.g. comparing different activations 
    by just changing `model.params.activation` while keeping the same CLI args).
    """
    if ctx.in_dim is None:
        raise ValueError("ExampleMLP needs ctx.in_dim")

    model_params = getattr(args, "model_params", None)
    params = dict(model_params) if isinstance(model_params, Mapping) else {}

    # Simple helper to pick param values with priority: model_params > flat args > default
    def _pick(name: str, default):
        if name in params:
            return params[name]
        value = getattr(args, name, default)
        return default if value is None else value

    # Params with flexible naming and defaults
    hidden = int(_pick("hidden", 128))
    layers = int(_pick("layers", 2))
    activation = str(_pick("activation", _pick("hidden_activation", "relu")))
    return ExampleMLP(
        in_dim=ctx.in_dim,
        hidden=hidden,
        num_classes=ctx.num_classes,
        num_layers=layers,
        activation=activation,
    )




# THIS PART IS NOT NEEDED FOR NORMAL USAGE of the model (Using backpropagation as learning rule), 
# but just for demonstrating how to use the autocache framework with different CacheSpec settings, 
# to help build alternative learning rules.
if __name__ == "__main__":
    model = ExampleMLP(in_dim=10, hidden=32, num_classes=5, num_layers=2, activation="relu")
    x = torch.randn(4, 10)

    logits = model(x)
    print("Output shape:", logits.shape)

    # Example A: param-only cache 
    logits, cache, views = forward_with_standard_cache(
        model,
        x,
        cache_spec=CacheSpec(
            param_module_types=(nn.Linear,),
            require_block_inputs=True,
            require_block_outputs=True,
            require_single_call=True,
            require_single_output_head=True,
        ),
    )

    print("\n[Param-only cache]")
    print("Selected blocks:", [b.name for b in views["selected_blocks"]])
    print("Param blocks:", [b.name for b in views["param_blocks"]])
    print("Output block:", [b.name for b in views["output_blocks"]])
    print("Hidden blocks:", [b.name for b in views["hidden_blocks"]])
    print("Cache keys:", sorted(cache.keys()))

    # local_blocks expose x/u/h in a rule-friendly structure
    for lb in views["local_blocks"]:
        x_shape = tuple(lb["x"].shape) if torch.is_tensor(lb["x"]) else None
        u_shape = tuple(lb["u"].shape) if torch.is_tensor(lb["u"]) else None
        h_shape = tuple(lb["h"].shape) if torch.is_tensor(lb["h"]) else None
        print(
            f"local_block name={lb['name']} is_output={lb['is_output']} "
            f"x={x_shape} u={u_shape} h={h_shape}"
        )

    # Example B: activation pairing (u -> h) with module activations
    _logits, _cache, paired_views = forward_with_standard_cache(
        model,
        x,
        cache_spec=CacheSpec(
            observed_module_types=(nn.Linear, nn.ReLU),
            param_module_types=(nn.Linear,),
            activation_module_types=(nn.ReLU,),
            require_block_inputs=True,
            require_block_outputs=True,
            require_single_call=True,
            require_single_output_head=True,
            require_activation_pairing=True,
        ),
    )
    print("\n[Activation pairing]")
    for lb in paired_views["local_blocks"]:
        if lb["is_output"]:
            continue
        h_shape = tuple(lb["h"].shape) if torch.is_tensor(lb["h"]) else None
        print(f"paired hidden block={lb['name']} post_act_shape={h_shape}")
