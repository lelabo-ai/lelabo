# lab/models/mlp.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .blocks import BlockModel, BlockSpec
from .registry import register_model, ModelContext


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


@dataclass
class MLPCache:
    # per-layer aligned indexing:
    # inputs[i]  = x_in to linear[i]
    # preacts[i] = u = linear[i](x_in)
    # acts[i]    = output after nonlinearity for hidden layers; for head we store logits as acts[-1]
    inputs: List[torch.Tensor]
    preacts: List[torch.Tensor]
    acts: List[torch.Tensor]

    def as_dict(self) -> Dict[str, List[torch.Tensor]]:
        return {"inputs": self.inputs, "preacts": self.preacts, "acts": self.acts}


def forward_mlp_with_cache(
    linears: nn.ModuleList,
    x: torch.Tensor,
    act: Callable[[torch.Tensor], torch.Tensor],
    *,
    return_cache: bool,
) -> Tuple[torch.Tensor, Optional[Dict[str, List[torch.Tensor]]]]:
    if not return_cache:
        a = x
        for i in range(len(linears) - 1):
            a = act(linears[i](a))
        logits = linears[-1](a)
        return logits, None

    cache = MLPCache(inputs=[], preacts=[], acts=[])
    a = x

    # hidden layers
    for i in range(len(linears) - 1):
        cache.inputs.append(a)          # x_in
        u = linears[i](a)               # preact / current
        cache.preacts.append(u)         # u
        a = act(u)                      # post-act
        cache.acts.append(a)            # store post-act aligned with layer i

    # head
    cache.inputs.append(a)              # x_in to head
    logits = linears[-1](a)             # u for head
    cache.preacts.append(logits)
    cache.acts.append(logits)           # store logits as "act" for head (keeps lengths aligned)

    return logits, cache.as_dict()


class MLPStack(nn.Module):
    def __init__(self, dims: List[int], activation: str = "tri"):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output")
        self.dims = [int(d) for d in dims]
        self.activation = str(activation).lower()
        self.act = make_activation(self.activation)
        self.linears = nn.ModuleList(
            [nn.Linear(self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)]
        )

    @property
    def head(self) -> nn.Linear:
        return self.linears[-1]

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        logits, cache = forward_mlp_with_cache(self.linears, x, self.act, return_cache=return_cache)
        if not return_cache:
            return logits

        # Build the exact keys expected by SoftHebbDemoConv:
        # cache["block_inputs"][name] = x_in
        # cache["block_outputs"][name] = u
        block_inputs: Dict[str, torch.Tensor] = {}
        block_outputs: Dict[str, torch.Tensor] = {}

        L = len(self.linears)
        # i runs 0..L-1, aligned with cache.inputs/preacts
        for i in range(L):
            name = f"layer{i}" if i < (L - 1) else "head"
            block_inputs[name] = cache["inputs"][i]      # x_in
            block_outputs[name] = cache["preacts"][i]    # u

        cache["block_inputs"] = block_inputs
        cache["block_outputs"] = block_outputs
        return logits, cache


@register_model("mlp")
def build_mlp(ctx: ModelContext, args):
    if ctx.in_dim is None:
        raise ValueError("MLP needs ctx.in_dim")
    return MLPClassifier(
        in_dim=ctx.in_dim,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=ctx.num_classes,
    )


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
