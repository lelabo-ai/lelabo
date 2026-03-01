from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import ModelContext, register_model


def make_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = str(name).lower()
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    if name == "gelu":
        return F.gelu
    if name in ("silu", "swish"):
        return F.silu
    if name == "heaviside":
        return lambda x: (x >= 0).to(x.dtype)
    if name == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    if name in ("triangle", "tri"):

        def triangle(u: torch.Tensor) -> torch.Tensor:
            if u.dim() == 2:
                mu = u.mean(dim=1, keepdim=True)
                return F.relu(u - mu)
            if u.dim() == 4:
                mu = u.mean(dim=1, keepdim=True)
                return F.relu(u - mu)
            mu = u.mean(dim=-1, keepdim=True)
            return F.relu(u - mu)

        return triangle
    raise ValueError(f"Unknown activation: {name}")


class MLPStack(nn.Module):
    """Simple MLP stack with explicit layer names: layer0..layerN + head."""

    def __init__(self, dims: list[int], activation: str = "tri"):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output")

        self.dims = [int(d) for d in dims]
        self.activation = str(activation).lower()
        self.act = make_activation(self.activation)

        self._linear_names: list[str] = []
        last_idx = len(self.dims) - 2
        for i in range(last_idx + 1):
            name = "head" if i == last_idx else f"layer{i}"
            layer = nn.Linear(self.dims[i], self.dims[i + 1])
            setattr(self, name, layer)
            self._linear_names.append(name)

    @property
    def linears(self) -> list[nn.Linear]:
        return [getattr(self, name) for name in self._linear_names]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for name in self._linear_names[:-1]:
            h = self.act(getattr(self, name)(h))
        return getattr(self, self._linear_names[-1])(h)


@register_model("mlp")
def build_mlp(ctx: ModelContext, args):
    if ctx.in_dim is None:
        raise ValueError("MLP needs ctx.in_dim")

    model_params = getattr(args, "model_params", None)
    params = dict(model_params) if isinstance(model_params, Mapping) else {}

    def _pick(name: str, default):
        if name in params:
            return params[name]
        value = getattr(args, name, default)
        return default if value is None else value

    activation = _pick("activation", _pick("hidden_activation", _pick("hidden_act", "relu")))
    return MLPClassifier(
        in_dim=ctx.in_dim,
        hidden_dim=int(_pick("hidden", 2048)),
        num_layers=int(_pick("layers", 4)),
        num_classes=ctx.num_classes,
        activation=str(activation),
    )


class MLPClassifier(nn.Module):
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
        self.net = MLPStack(dims, activation=self.activation)

    @property
    def linears(self) -> list[nn.Linear]:
        return self.net.linears

    @property
    def head(self) -> nn.Linear:
        return self.net.head

    @property
    def act(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.net.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MLP = MLPClassifier
