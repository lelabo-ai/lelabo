# lab/models/qnet.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp_utils import forward_mlp_with_cache


class QNet(nn.Module):
    """
    Q-network (DQN) pour obs vectorielles.
    forward(x) -> Q-values [B, A]
    Optionnel: return_cache=True -> (Q, cache) avec le même format que MLPClassifier.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        obs_dim = int(obs_dim)
        n_actions = int(n_actions)
        hidden = int(hidden)
        layers = int(layers)
        assert layers >= 1

        dims = [obs_dim] + [hidden] * layers + [n_actions]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

        # head explicite (utile pour heads-only)
        self.q_head = self.linears[-1]

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        def relu(a): return F.relu(a)
        q, cache = forward_mlp_with_cache(self.linears, x, relu, return_cache=return_cache)
        return (q, cache) if return_cache else q
