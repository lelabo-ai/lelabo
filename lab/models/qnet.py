# lab/models/qnet.py
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import BlockModel, BlockSpec
from .mlp import MLPStack


class QNet(BlockModel):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, layers: int = 2, activation: str = "relu"):
        super().__init__()
        dims = [int(obs_dim)] + [int(hidden)] * int(layers) + [int(n_actions)]
        self.activation = str(activation).lower()
        self.net = MLPStack(dims, activation=self.activation)

        self.linears = self.net.linears
        self.q_head: nn.Linear = self.net.head

    def get_blocks(self) -> list[BlockSpec]:
        blocks: list[BlockSpec] = []
        for i, lin in enumerate(self.linears):
            is_out = (i == (len(self.linears) - 1))
            name = f"layer{i}" if not is_out else "q_head"
            blocks.append(BlockSpec(name=name, module=lin, rep="identity", is_output=is_out))
        return blocks

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        return self.net(x, return_cache=return_cache)
