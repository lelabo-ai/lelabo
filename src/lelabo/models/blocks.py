from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn


InSelect = Callable[[tuple[Any, ...], Optional[dict[str, Any]]], Any]
OutSelect = Callable[[Any], Any]


@dataclass(frozen=True)
class BlockSpec:
    """Describe a model block used by local update rules and cache providers."""

    name: str
    module: nn.Module
    rep: str = "identity"
    is_output: bool = False
    group: str = "main"
    params: Optional[Sequence[nn.Parameter]] = None
    in_select: Optional[InSelect] = None
    out_select: Optional[OutSelect] = None

    def iter_params(self):
        if self.params is not None:
            yield from self.params
        else:
            yield from self.module.parameters()


def normalize_standard_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Normalize cache payload to the framework standard schema."""
    if not isinstance(cache, dict):
        raise TypeError("Cache payload must be a dict.")

    block_inputs = cache.get("block_inputs", {})
    block_outputs = cache.get("block_outputs", {})
    if not isinstance(block_inputs, dict) or not isinstance(block_outputs, dict):
        raise TypeError("Cache must contain dict keys: 'block_inputs' and 'block_outputs'.")

    out = dict(cache)
    out["cache_version"] = str(cache.get("cache_version", "standard.v1"))
    out["block_inputs"] = block_inputs
    out["block_outputs"] = block_outputs
    # v1 keeps only neutral output naming. Pre-activations can be added in a later version.
    out.pop("block_preacts", None)
    out.pop("block_preacts_all", None)

    block_inputs_all = out.get("block_inputs_all")
    if not isinstance(block_inputs_all, dict):
        out["block_inputs_all"] = {k: [v] for k, v in block_inputs.items() if torch.is_tensor(v)}

    block_outputs_all = out.get("block_outputs_all")
    if not isinstance(block_outputs_all, dict):
        out["block_outputs_all"] = {k: [v] for k, v in block_outputs.items() if torch.is_tensor(v)}

    return out
