# Model Cache Advanced

This file is only for advanced model work.

For a normal BP model plugin, `models/example.py` is enough.

Use this file only when you need:

- local rules
- block-level activations
- explicit cache inspection
- model compatibility with cache-based update rules

## When a simple model is enough

A normal supervised BP path only needs:

- a registered model builder
- an `nn.Module`
- a normal `forward(...)`

No cache provider work is required.

## When to think about cache provider / blocks

You need cache-aware design when an update rule or analysis depends on intermediate activations.

Typical examples:

- output-head-only local rules
- blockwise learning rules
- research diagnostics over internal blocks

## Core pieces

- `forward_with_standard_cache(...)`
- `CacheSpec`
- output blocks
- declared blocks

See `models/cache_walkthrough.py` for a runnable Python reference.

## `CacheSpec`

`CacheSpec` declares what to record during the forward pass:

- which module types are trainable
- which module types are observed
- whether to capture inputs
- whether to capture outputs
- whether the model must expose a single output head

This is what lets an update rule recover activations without rewriting the model forward pass.

## Output blocks vs declared blocks

- `output_blocks`
  Usually the blocks treated as output heads by the cache provider

- `declared_blocks`
  Blocks explicitly exposed by the model for structured access

Different update rules may rely on one or the other.

## Implications for local rules

If a local rule expects:

- a single Linear output head
- a specific hidden block shape
- captured input activations

then the model must expose those structures cleanly.

If not, the rule may fail even though BP training works.

## Recommended workflow

1. Start with the simplest model possible
2. Validate BP first
3. Only then add cache/block compatibility if the update rule requires it
4. Use `models/cache_walkthrough.py` as the reference implementation pattern

## Common failure modes

- Model trains with BP but local rule fails
  The expected blocks or cached tensors are missing.

- Output head not found
  The cache spec likely expects a single head that the model does not expose clearly.

- Wrong tensor shape in the rule
  The captured block inputs/outputs do not match the rule’s assumptions.
