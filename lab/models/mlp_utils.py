# lab/models/mlp_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    elif name=='softmax':
        return lambda x: F.softmax(x, dim=-1)
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class MLPCache:
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
    cache.acts.append(a)

    for i in range(len(linears) - 1):
        cache.inputs.append(a)
        z = linears[i](a)
        cache.preacts.append(z)
        a = act(z)
        cache.acts.append(a)

    cache.inputs.append(a)
    logits = linears[-1](a)
    cache.preacts.append(logits)
    return logits, cache.as_dict()


class MLPStack(nn.Module):
    def __init__(self, dims: List[int], activation: str = "relu"):
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

        block_inputs = {}
        block_outputs = {}
        for i in range(len(cache["inputs"])):
            name = f"layer{i}" if i < (len(cache["inputs"]) - 1) else "head"
            block_inputs[name] = cache["inputs"][i]
            block_outputs[name] = cache["preacts"][i]
        cache["block_inputs"] = block_inputs
        cache["block_outputs"] = block_outputs
        return logits, cache
