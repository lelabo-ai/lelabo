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

from lab.models.blocks import LeModule, BlockSpec
from lab.models.registry import register_model, ModelContext


# ============================================================
# 1️⃣ Model definition (inherits from LeModule)
# ============================================================

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


# ============================================================
# 2️⃣ Model registry
# ============================================================

@register_model("example_mlp")
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


# ============================================================
# 3️⃣ Optional: custom block definition
# ============================================================

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
