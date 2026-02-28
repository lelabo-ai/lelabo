from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .blocks import normalize_standard_cache
from .cache_wrapper import ModelCacheWrapper


class ContractError(RuntimeError):
    """Raised when cache/model contract requirements are not satisfied."""


@dataclass(frozen=True)
class CacheSpec:
    require_block_inputs: bool = True
    require_block_outputs: bool = False
    require_single_call: bool = False
    require_single_output: bool = False
    require_linear_only: bool = False
    require_ndim2_inputs: bool = False


def _normalize_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_standard_cache(dict(cache))


def _collect_blocks(model, cache: Mapping[str, Any]) -> list[Any]:
    runtime = cache.get("block_specs_runtime", None)
    if isinstance(runtime, list) and runtime:
        return runtime
    if hasattr(model, "get_blocks"):
        try:
            blocks = model.get_blocks()
            if isinstance(blocks, list):
                return blocks
        except Exception:
            pass
    return []


def _validate_spec(cache: Mapping[str, Any], blocks: list[Any], spec: CacheSpec) -> None:
    if spec.require_block_inputs and not isinstance(cache.get("block_inputs"), dict):
        raise ContractError("Cache missing required key: 'block_inputs'.")
    if spec.require_block_outputs and not isinstance(cache.get("block_outputs"), dict):
        raise ContractError("Cache missing required key: 'block_outputs'.")

    if spec.require_single_output:
        n_out = sum(bool(getattr(b, "is_output", False)) for b in blocks)
        if n_out != 1:
            raise ContractError(f"CacheSpec requires exactly one output block, got {n_out}.")

    if spec.require_linear_only:
        non_linear = [
            str(getattr(b, "name", "<unnamed>"))
            for b in blocks
            if not isinstance(getattr(b, "module", None), nn.Linear)
        ]
        if non_linear:
            raise ContractError(
                "CacheSpec requires linear-only blocks, found unsupported blocks: "
                + ", ".join(non_linear)
            )

    if spec.require_single_call:
        all_inputs = cache.get("block_inputs_all", {})
        if isinstance(all_inputs, Mapping):
            multi = [name for name, vals in all_inputs.items() if isinstance(vals, list) and len(vals) != 1]
            if multi:
                raise ContractError(
                    "CacheSpec requires each block to be called exactly once; "
                    f"violations: {multi}"
                )

    if spec.require_ndim2_inputs:
        block_inputs = cache.get("block_inputs", {})
        if isinstance(block_inputs, Mapping):
            bad = []
            for name, value in block_inputs.items():
                if torch.is_tensor(value) and value.dim() == 2:
                    continue
                bad.append(f"{name}:{getattr(value, 'shape', None)}")
            if bad:
                raise ContractError(
                    "CacheSpec requires 2D inputs for all selected blocks. Offending blocks: "
                    + ", ".join(bad)
                )


def forward_with_standard_cache(model, *args, cache_spec: CacheSpec | None = None, **kwargs):
    spec = cache_spec or CacheSpec()

    explicit_specs = None
    if not isinstance(model, ModelCacheWrapper) and hasattr(model, "get_blocks"):
        try:
            candidate = model.get_blocks()
            if isinstance(candidate, list) and candidate:
                explicit_specs = candidate
        except Exception:
            explicit_specs = None

    wrapper = model if isinstance(model, ModelCacheWrapper) else ModelCacheWrapper(model, block_specs=explicit_specs)
    out, cache_raw = wrapper(*args, return_cache=True, **kwargs)
    cache = _normalize_cache(cache_raw)
    blocks = _collect_blocks(wrapper, cache)
    _validate_spec(cache, blocks, spec)
    return out, cache, blocks
