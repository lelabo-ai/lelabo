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

    module_inputs = cache.get("module_inputs", {})
    module_outputs = cache.get("module_outputs", {})
    if not isinstance(module_inputs, dict) or not isinstance(module_outputs, dict):
        raise TypeError("Cache must contain dict keys: 'module_inputs' and 'module_outputs'.")

    out = dict(cache)
    out["cache_version"] = str(cache.get("cache_version", "standard.v1"))
    out["module_inputs"] = module_inputs
    out["module_outputs"] = module_outputs

    module_inputs_all = out.get("module_inputs_all")
    if not isinstance(module_inputs_all, dict):
        module_inputs_all = {k: [v] for k, v in module_inputs.items() if torch.is_tensor(v)}
    out["module_inputs_all"] = module_inputs_all

    module_outputs_all = out.get("module_outputs_all")
    if not isinstance(module_outputs_all, dict):
        module_outputs_all = {k: [v] for k, v in module_outputs.items() if torch.is_tensor(v)}
    out["module_outputs_all"] = module_outputs_all

    call_count = out.get("call_count_by_name")
    if not isinstance(call_count, dict):
        call_count = {}
        for name, vals in module_outputs_all.items():
            if isinstance(vals, list):
                call_count[str(name)] = int(len(vals))
    out["call_count_by_name"] = call_count

    # v2 keeps only neutral output naming. Pre-activations can be added in a later version.
    out.pop("block_preacts", None)
    out.pop("block_preacts_all", None)
    out.pop("block_inputs", None)
    out.pop("block_outputs", None)
    out.pop("block_inputs_all", None)
    out.pop("block_outputs_all", None)

    return out
