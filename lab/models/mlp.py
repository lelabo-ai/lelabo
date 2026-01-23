# lab/models/mlp.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp_utils import forward_mlp_with_cache


class MLPClassifier(nn.Module):
    """
    MLP pour classification.
    Cache (return_cache=True) :
      {"inputs": [...], "preacts": [...], "acts": [...]}
    Compatible TargetPropagation (model.linears + return_cache=True).
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)
        self.activation = str(activation).lower()

        dims = [self.in_dim] + [self.hidden_dim] * self.num_layers + [self.num_classes]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "heaviside":
            return (x >= 0).to(x.dtype)
        raise ValueError(f"Unknown activation: {self.activation}")

    def act_deriv_from_preact(self, z: torch.Tensor) -> torch.Tensor:
        # (si tu utilises TP/STE heaviside ailleurs, tu peux l'étendre ici)
        if self.activation == "relu":
            return (z > 0).to(z.dtype)
        if self.activation == "tanh":
            # deriv tanh = 1 - tanh(z)^2 ; ici on n'a que z => ok
            t = torch.tanh(z)
            return 1.0 - t * t
        if self.activation == "sigmoid":
            s = torch.sigmoid(z)
            return s * (1.0 - s)
        if self.activation == "heaviside":
            # par défaut: dérivée nulle (ou STE si tu veux)
            return torch.zeros_like(z)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        logits, cache = forward_mlp_with_cache(self.linears, x, self.act, return_cache=return_cache)
        return (logits, cache) if return_cache else logits
