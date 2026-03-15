# Model Cache Advanced

This file is only for advanced model work.

For a normal BP model plugin, `models/example.py` is enough. Do not open this
file unless your extension needs intermediate activations, block semantics, or
cache-aware local rules.

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

## When to think about cache / blocks

You need cache-aware design when a rule or analysis depends on intermediate activations.

Typical examples:

- blockwise learning rules
- local rules with explicit hidden-block updates
- probing or diagnostics over internal blocks

## Stable public pieces

- `declare_blocks()`
- `BlockSpec`
- `ResolvedBlock`
- `CacheSpec`
- `forward_with_standard_cache(...)`
- `declared`
- `execution`
- `paired_execution`
- `register_cache_pair_activation(...)`

These are the parts you can reasonably depend on in capsule code.

## Noble vs universal path

Noble path:

- your model implements `declare_blocks()`
- the runtime can expose `views["declared"]`
- use this when the model has meaningful semantic blocks
- `declare_blocks()` is optional unless that semantic block structure matters

Universal path:

- LeLabo auto-observes `nn.Module`s
- the runtime can expose `views["execution"]`
- use this when the model is a regular PyTorch module and you do not want custom block declarations

Paired universal path:

- LeLabo starts from `execution`
- then enriches it with post-activation pairing
- this becomes `views["paired_execution"]`

## `BlockSpec`

`BlockSpec` is declarative.

Important fields:

- `name`
- `module`
- `rep`
- `is_output`
- `group`
- `params`
- `in_select`
- `out_select`

`BlockSpec` describes a block. It is not the runtime block object.

## `ResolvedBlock`

`ResolvedBlock` is the runtime block you receive in the selected cache view.

Important fields:

- `name`
- `module`
- `spec`
- `rep`
- `group`
- `is_trainable`
- `is_output`
- `x`
- `u`
- `h`
- `activation_name`
- `exec_module`
- `exec_span_names`

### Meaning of the main tensors

- `x`
  best available input tensor for that block

- `u`
  best available output / pre-activation-side tensor for that block

- `h`
  post-activation tensor when pairing is available

### `exec_module`

`exec_module` is the best local replay module the runtime can expose for that runtime block.

Use it for:

- local segment replay
- rules that need a block-like executable object

`exec_span_names` tells you which runtime names are covered by that replay module.

## `CacheSpec`

`CacheSpec` tells the runtime what to capture and what to validate.

Important fields:

- `target_view`
- `trainable_module_types`
- `observed_module_types`
- `observed_module_names`
- `capture_inputs`
- `capture_outputs`
- `require_single_call`
- `require_single_output_head`
- `require_input_ndim`
- `require_output_ndim`

Important runtime rule:

- `forward_with_standard_cache(...)` returns `out`, `cache`, and `views`
- `views` contains only the requested `CacheSpec.target_view`, not all public views at once

## Custom activation pairing

If your model uses a custom activation module, register it:

```python
from lelabo.models import register_cache_pair_activation
```

This allows `paired_execution` to discover post-activation tensors for that module family.

## Public stable vs runtime detail

Public stable contract:

- `declare_blocks()`
- `BlockSpec`
- `ResolvedBlock`
- `CacheSpec`
- `declared`
- `execution`
- `paired_execution`
- `register_cache_pair_activation(...)`

Runtime detail, not a stable API:

- the exact pairing heuristic
- internal replay reconstruction helpers
- the exact contents of `cache["_runtime"]`

Do not make capsule code depend on internal `_runtime` shapes unless you are deliberately doing local research on internals.

## Current limits

What works well:

- standard supervised training
- many local-rule setups based on intermediate activations
- cache-aware models that either declare their blocks or expose regular `nn.Module` structure

What is not first-class yet:

- general staged multi-phase training programs
- native pretrain/freeze/fine-tune programs as public runtime objects
- arbitrary papers with several loaders, phases, and optimizers expressed as a first-class experiment program

Some papers can still be implemented, but part of the orchestration may still live in the rule or in the experiment setup.

## Recommended workflow

1. Start with the simplest model possible
2. Validate BP first
3. Add cache/block compatibility only if the rule needs it
4. Use `models/cache_walkthrough.py` as the runnable reference pattern
