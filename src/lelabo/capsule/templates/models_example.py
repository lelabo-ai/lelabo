"""
example.py

Minimal documented example showing:

1) How to register a model
2) How to inherit from LeModule
3) How automatic block discovery works
4) How to use return_cache=True
5) How to optionally override get_blocks()
"""

from __future__ import annotations
import torch
import torch.nn as nn

from lelabo.models.blocks import LeModule, BlockSpec
from lelabo.models.registry import register_model, ModelContext


# Model definition (inherits from LeModule)
# LeModules provide automatic block discovery and cache collection to become usable for update rules.
# If you don't need these features, you can inherit from nn.Module instead. But we strongly recommend 
# inheriting from LeModule for better integration with the rest of the framework.

class ExampleMLP(LeModule):
    """
    Minimal MLP example.

    Because we inherit from LeModule:
    - Linear layers are automatically treated as blocks
    - return_cache=True automatically collects activations
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Model registry

# To make this model available in the command line interface, we need to register it.
# Please uncomment the @register_model decorator and implement the build_example_mlp function.

#@register_model("example_mlp")
def build_example_mlp(ctx: ModelContext, args):
    """
    Factory used by:

        lelabo train supervised --dataset iris --model example_mlp

    ctx provides:
        - ctx.in_dim
        - ctx.num_classes
        - dataset-dependent information
    """

    hidden = getattr(args, "hidden", 128)

    return ExampleMLP(
        in_dim=ctx.in_dim,
        hidden=hidden,
        num_classes=ctx.num_classes,
    )


# Optional: custom block definition to override automatic block discovery
# This is only needed if you want to have fine control over what is considered a block and what is not.
# Particularly if you want more granular control over the cache collection (e.g. you want to collect activations 
# from non-linearities or other operations that are not automatically considered blocks).
# Or you want to build an update rules that relies on cache that we (for the moment) don't 
# automatically collect. We encourage you to create an issues if you have a use case that 
# is not covered by the automatic block discovery and cache collection, so we can add it to the framework.

class CustomBlockMLP(LeModule):
    """
    Same idea as ExampleMLP,
    but we explicitly define blocks.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.head(h)

    def get_blocks(self):
        """
        Override automatic block discovery.

        Only do this if you need fine control.
        """
        return [
            BlockSpec(
                name="encoder",
                module=self.encoder,
                is_output=False,
            ),
            BlockSpec(
                name="head",
                module=self.head,
                is_output=True,
            ),
        ]


# ============================================================
# 4️⃣ Standalone usage example
# ============================================================

if __name__ == "__main__":
    model = ExampleMLP(in_dim=10, hidden=32, num_classes=5)
    x = torch.randn(4, 10)

    # Normal forward
    logits = model(x)

    # Forward with automatic cache collection
    logits, cache = model(x, return_cache=True)

    print("Output shape:", logits.shape)
    print("Blocks:", list(cache["block_outputs"].keys()))
    print("Cache keys:", list(cache.keys()))
