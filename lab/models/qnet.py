# models/qnet.py
from __future__ import annotations
import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        assert layers >= 1

        modules = []
        in_dim = obs_dim
        for _ in range(layers):
            modules.append(nn.Linear(in_dim, hidden))
            modules.append(nn.ReLU())
            in_dim = hidden
        modules.append(nn.Linear(hidden, n_actions))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
