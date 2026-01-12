# lab/models/mlp_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import torch
import torch.nn as nn


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
    """
    Forward d'un MLP défini par une liste de Linear.
    - applique act() après chaque layer sauf la dernière (logits/head)
    - cache standard: inputs/preacts/acts
    """
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
