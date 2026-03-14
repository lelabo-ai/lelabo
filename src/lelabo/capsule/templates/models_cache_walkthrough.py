"""
cache_walkthrough.py

Reference-only notes for advanced model work.

This file is not auto-loaded by LeLabo because it does not register anything.
Keep `models/example.py` simple. Use this file only when you need block-level caches
for local rules, diagnostics, or research tooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache


def example_cache_spec() -> CacheSpec:
    """
    Minimal cache spec that captures an execution view over Linear blocks.

    This is useful when an update rule needs intermediate activations.
    """
    return CacheSpec(
        target_view="execution",
        trainable_module_types=(nn.Linear,),
        observed_module_types=(nn.Linear,),
        capture_inputs=True,
        capture_outputs=True,
        require_single_output_head=True,
    )


@torch.no_grad()
def inspect_output_blocks(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, list[object]]:
    """
    Run a forward pass and return the execution-view blocks marked as outputs.
    """
    logits, _cache, views = forward_with_standard_cache(
        model,
        x,
        cache_spec=example_cache_spec(),
    )
    execution_blocks = views.get("execution", [])
    if not isinstance(execution_blocks, list):
        execution_blocks = []
    output_blocks = [block for block in execution_blocks if bool(block.get("is_output", False))]
    return logits, output_blocks


def notes() -> str:
    return (
        "Use `forward_with_standard_cache(...)` only when a local rule or analysis really needs it. "
        "Remember that the runtime returns only the requested target view. "
        "For a standard BP model plugin, keep `models/example.py` simple."
    )
