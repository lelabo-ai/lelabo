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
    Minimal cache spec that captures Linear blocks and their inputs/outputs.

    This is useful when an update rule needs intermediate activations.
    """
    return CacheSpec(
        trainable_module_types=(nn.Linear,),
        observed_module_types=(nn.Linear,),
        capture_inputs=True,
        capture_outputs=True,
        require_single_output_head=True,
    )


@torch.no_grad()
def inspect_output_blocks(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """
    Run a forward pass and return the output blocks recorded by the cache provider.
    """
    logits, _cache, views = forward_with_standard_cache(
        model,
        x,
        cache_spec=example_cache_spec(),
    )
    output_blocks = views.get("output_blocks", [])
    if not isinstance(output_blocks, list):
        output_blocks = []
    return logits, output_blocks


def notes() -> str:
    return (
        "Use `forward_with_standard_cache(...)` only when a local rule or analysis really needs it. "
        "For a standard BP model plugin, keep `models/example.py` simple."
    )
