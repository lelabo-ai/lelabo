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
            # Enforce that exactly one output head is inferred (last trainable observed block)
            require_single_output_head=True,
            # Optional expected ndim for trainable block inputs (None disables the check)
            require_input_ndim=2,
            # Optional expected ndim for trainable block outputs (None disables the check)
            require_output_ndim=2,
            # Build rule-friendly local replay blocks from trainable segments
            include_local_blocks=True,
            # Try to auto-pair trainable block output with next activation output
            auto_pair_post_activation=True,
            # Activation module types used for auto-pairing when enabled
            # if None, the activation types will be determined by torch default activations (torch.nn.Relu, torch.nn.LeakyRelu, ...)
            # This parameter is actually very useful to enable auto-pairing for non-standard activations (e.g. Swish, Triangle) by just
            # adding the activation types here, without needing to change the model code or use explicit module naming.
            auto_pair_activation_types=(nn.ReLU, nn.Tanh, nn.Sigmoid),
            # Include model-defined pre-cut blocks from model.get_blocks() when available
            # This is particularly useful if you want to have more control over the block definitions
            # Or you want for ResNet/Transformer style models where the natural block definitions are not strictly module-based.
            include_model_blocks=True,
        ),
    )


    # import utils to print cache views in a readable way
    from lelabo.capsule.templates.print_utils import (
        print_blocks_table,
        print_local_block_details,
        print_local_blocks_table,
    )

    print("\n=== Param-only cache ===")
    print_blocks_table("Ordered blocks", views["ordered_blocks"])
    output_block = views["output_block"]
    print("Output block:", output_block["name"] if isinstance(output_block, dict) else "<none>")
    print_blocks_table("Model blocks (if model.get_blocks())", views["model_blocks"])
    print_local_blocks_table(views["local_blocks"])
    print("Cache keys:", sorted(cache.keys()))

    # Example B: inspect the cache of an existing builtin CNN model
    from lelabo.models.builtins.convnet import ConvNetClassifier

    cnn = ConvNetClassifier(
        in_channels=3,
        num_classes=10,
        channels=[16, 32],
        use_bn=True,
        activation="relu",
    )
    x_cnn = torch.randn(2, 3, 32, 32)

    cnn_logits, cnn_cache, cnn_views = forward_with_standard_cache(
        cnn,
        x_cnn,
        cache_spec=CacheSpec(
            trainable_module_types=(nn.Conv2d, nn.Linear),
            # `observed_module_types` is omitted on purpose:
            # the provider will observe all leaf modules by default
            # (Conv/BN/Act/Pool/Linear), which is often what local rules want.
            require_single_call=True,
            require_single_output_head=True,
            include_local_blocks=True,
        ),
    )

    print("\n=== Builtin CNN cache example ===")
    print(cnn)
    print("CNN input shape:", tuple(x_cnn.shape))
    print("CNN output shape:", tuple(cnn_logits.shape))
    print_blocks_table("CNN ordered blocks", cnn_views["ordered_blocks"])
    cnn_output_block = cnn_views["output_block"]
    print("CNN output block:", cnn_output_block["name"] if isinstance(cnn_output_block, dict) else "<none>")
    print_local_blocks_table(cnn_views["local_blocks"])
    print_local_block_details(cnn_views["local_blocks"])
    print("CNN cache keys:", sorted(cnn_cache.keys()))
