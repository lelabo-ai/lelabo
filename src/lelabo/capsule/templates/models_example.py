"""
example.py

Minimal documented example showing:

1) How to register a model
2) How to keep model API pure nn.Module
3) How to optionally expose get_blocks()
4) How to collect cache from the framework side
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lelabo.models.blocks import BlockSpec
from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache
from lelabo.models.registry import ModelContext, register_model


class ExampleMLP(nn.Module):
    """Minimal MLP example with standard PyTorch API."""

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


# @register_model("example_mlp")
def build_example_mlp(ctx: ModelContext, args):
    hidden = getattr(args, "hidden", 128)
    return ExampleMLP(in_dim=ctx.in_dim, hidden=hidden, num_classes=ctx.num_classes)


class CustomBlockMLP(nn.Module):
    """Same idea as ExampleMLP, but with explicit block contract."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        return self.head(h)

    def get_blocks(self):
        return [
            BlockSpec(name="encoder", module=self.encoder, is_output=False),
            BlockSpec(name="head", module=self.head, is_output=True),
        ]


if __name__ == "__main__":
    model = ExampleMLP(in_dim=10, hidden=32, num_classes=5)
    x = torch.randn(4, 10)

    logits = model(x)
    print("Output shape:", logits.shape)

    # Cache is collected by the framework provider, not by model inheritance.
    logits, cache, blocks = forward_with_standard_cache(
        model,
        x,
        cache_spec=CacheSpec(require_block_inputs=True, require_block_outputs=True),
    )

    print("Detected blocks:", [b.name for b in blocks])
    print("Cached outputs:", list(cache["block_outputs"].keys()))
    print("Cache keys:", list(cache.keys()))
