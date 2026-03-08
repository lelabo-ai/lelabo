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

from lelabo.models.registry import ModelContext, build_model, register_model


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


# Uncomment this decorator to make the model visible to `lelabo list` / `lelabo train`.
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
    print("=== ExampleMLP ===")
    print(model)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(logits.shape))

    # Import cache spec and forward_with_standard_cache for demonstration
    from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache
    
    # Example A: param-only cache 
    logits, cache, views = forward_with_standard_cache(
        model,
        x,
        cache_spec=CacheSpec(
            # Which module types should be treated as trainable local blocks
            trainable_module_types=(nn.Linear,),
            # Which module types should be observed by runtime hooks
            observed_module_types=(nn.Linear,),
            # Optional explicit module names to observe (empty means \"use observed_module_types\")
            observed_module_names=(),
            # How declared blocks from model.get_blocks() are used:
            # "ignore" = auto-only selection, "merge" = declared + auto, "only" = declared only
            declared_blocks_mode="merge",
            # Store first-call input tensor per observed module in cache[\"module_inputs\"]
            capture_inputs=True,
            # Store first-call output tensor per observed module in cache[\"module_outputs\"]
            capture_outputs=True,
            # Store per-call tensor history in module_inputs_all/module_outputs_all
            capture_all_calls=True,
            # Store ordered execution trace (steps, ref ids, call indexes)
            capture_steps=True,
            # Enforce that each observed module is called exactly once per forward
            require_single_call=True,
            # Enforce that exactly one output head is selected
            require_single_output_head=True,
            # Optional expected ndim for trainable block inputs (None disables the check)
            require_input_ndim=2,
            # Optional expected ndim for trainable block outputs (None disables the check)
            require_output_ndim=2,
            # Try to auto-pair trainable block output with next activation output
            auto_pair_post_activation=True,
            # Activation module types used for auto-pairing when enabled
            # if None, the activation types will be determined by torch default activations (torch.nn.Relu, torch.nn.LeakyRelu, ...)
            # This parameter is actually very useful to enable auto-pairing for non-standard activations (e.g. Swish, Triangle) by just
            # adding the activation types here, without needing to change the model code or use explicit module naming.
            auto_pair_activation_types=(nn.ReLU, nn.Tanh, nn.Sigmoid),
        ),
    )


    # import utils to print cache views in a readable way
    from lelabo.capsule.templates.print_utils import (
        print_blocks_table,
        print_trainable_segment_details,
        print_trainable_segments_table,
    )

    print("\n=== Param-only cache ===")
    print_blocks_table("Execution blocks", views["execution_blocks"])
    output_blocks = views.get("output_blocks", [])
    print("Output blocks:", [b.get("name", "<unnamed>") for b in output_blocks if isinstance(b, dict)])
    print_blocks_table("Declared blocks (if model.get_blocks())", views["declared_blocks"])
    print_trainable_segments_table(views["trainable_segments"])
    print("Cache keys:", sorted(cache.keys()))

    # Example B: inspect the cache of a registered model by its registry key.
    # This example code is here to help the user understand the cache
    # of each registered model, and how to do the same for their own registered model.
    from types import SimpleNamespace
    
    
    if False: # Disable by default to avoid too much print, but you can enable it to inspect the cache of registered models in the lab or your own registered model.
        # You can change the name below to inspect the cache of each registered model
        # in the lab, and also to test the cache of your own registered model.
        registered_model_name = "cnn"
        
        registered_ctx = ModelContext(
            dataset="cifar10",
            num_classes=10,
            in_dim=None,
            in_channels=3,
            input_shape=(3, 32, 32),
        )
        registered_args = SimpleNamespace(
            model_params={
                "channels": [16, 32],
                "use_bn": True,
                "activation": "relu",
                "head_mode": "flatten",
            }
        )
        x_cnn = torch.randn(2, 3, 32, 32)
        registered_model = build_model(registered_model_name, registered_ctx, registered_args)

        cnn_logits, cnn_cache, cnn_views = forward_with_standard_cache(
            registered_model,
            x_cnn,
            cache_spec=CacheSpec(
                trainable_module_types=(nn.Conv2d, nn.Linear),
                # `observed_module_types` is omitted on purpose:
                # the provider will observe all leaf modules by default
                # (Conv/BN/Act/Pool/Linear), which is often what local rules want.
                require_single_call=True,
                require_single_output_head=True,
                auto_pair_post_activation=True,
            ),
        )

        print(f"\n=== Registered model cache example: {registered_model_name} ===")
        print(registered_model)
        print("CNN input shape:", tuple(x_cnn.shape))
        print("CNN output shape:", tuple(cnn_logits.shape))
        print_blocks_table("CNN execution blocks", cnn_views["execution_blocks"])
        cnn_output_blocks = cnn_views.get("output_blocks", [])
        print("CNN output blocks:", [b.get("name", "<unnamed>") for b in cnn_output_blocks if isinstance(b, dict)])
        print_trainable_segments_table(cnn_views["trainable_segments"])
        print_trainable_segment_details(cnn_views["trainable_segments"])
        print("CNN cache keys:", sorted(cnn_cache.keys()))
