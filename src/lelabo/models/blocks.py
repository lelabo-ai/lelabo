"""Block declaration helpers for cache-aware model execution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
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


@dataclass(frozen=True)
class ResolvedBlock(Mapping[str, Any]):
    """Resolved runtime block shared by declared and execution cache views."""

    name: str
    module: nn.Module
    spec: BlockSpec | None = None
    rep: str = "identity"
    group: str = "main"
    is_output: bool = False
    is_trainable: bool = False
    x: Any = None
    u: Any = None
    h: Any = None
    activation_name: str | None = None
    exec_module: nn.Module | None = None
    exec_span_names: tuple[str, ...] = ()
    call_count: int = 0

    @property
    def type(self) -> str:
        return self.module.__class__.__name__

    @property
    def available(self) -> bool:
        return bool(torch.is_tensor(self.x) or torch.is_tensor(self.u))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "type": self.type,
            "spec": self.spec,
            "rep": self.rep,
            "group": self.group,
            "is_output": self.is_output,
            "is_trainable": self.is_trainable,
            "x": self.x,
            "u": self.u,
            "h": self.h,
            "activation_name": self.activation_name,
            "exec_module": self.exec_module,
            "exec_span_names": self.exec_span_names,
            "call_count": self.call_count,
            "available": self.available,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


def normalize_standard_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Normalize cache payload to the framework standard schema."""
    if not isinstance(cache, dict):
        raise TypeError("Cache payload must be a dict.")

    module_inputs = cache.get("module_inputs", {})
    module_outputs = cache.get("module_outputs", {})
    if not isinstance(module_inputs, dict) or not isinstance(module_outputs, dict):
        raise TypeError("Cache must contain dict keys: 'module_inputs' and 'module_outputs'.")

    out = dict(cache)
    out["cache_version"] = str(cache.get("cache_version", "standard.v4"))
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

    runtime = out.get("_runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    if "steps" in out:
        runtime["steps"] = out.pop("steps")
    if "block_specs_runtime" in out:
        runtime["block_specs_runtime"] = out.pop("block_specs_runtime")
    if runtime:
        out["_runtime"] = runtime

    # v2 keeps only neutral output naming. Pre-activations can be added in a later version.
    out.pop("block_preacts", None)
    out.pop("block_preacts_all", None)
    out.pop("block_inputs", None)
    out.pop("block_outputs", None)
    out.pop("block_inputs_all", None)
    out.pop("block_outputs_all", None)

    return out
