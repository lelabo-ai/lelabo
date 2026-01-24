# lab/models/mlp.py
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .blocks import BlockModel, BlockSpec
    from .mlp_utils import MLPStack
except Exception:  # pragma: no cover
    from blocks import BlockModel, BlockSpec
    from mlp_utils import MLPStack


class MLPClassifier(BlockModel):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, activation: str = "relu"):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)
        self.activation = str(activation).lower()

        dims = [self.in_dim] + [self.hidden_dim] * self.num_layers + [self.num_classes]
        self.net = MLPStack(dims, activation=self.activation)
        self.linears = self.net.linears

    @property
    def head(self) -> nn.Linear:
        return self.net.head

    def get_blocks(self) -> list[BlockSpec]:
        blocks: list[BlockSpec] = []
        for i, lin in enumerate(self.linears):
            is_out = (i == (len(self.linears) - 1))
            name = f"layer{i}" if not is_out else "head"
            blocks.append(BlockSpec(name=name, module=lin, rep="identity", is_output=is_out))
        return blocks

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        return self.net(x, return_cache=return_cache)


MLP = MLPClassifier
