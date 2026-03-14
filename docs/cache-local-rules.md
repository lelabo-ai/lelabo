# Cache and Local Rules

This page documents the public cache/local-rule contract that LeLabo currently supports.

## Why this exists

Local learning rules need access to intermediate activations without forcing every model to adopt a custom framework base class.

LeLabo provides this through:

- `declare_blocks()` for a noble, semantic block declaration
- `forward_with_standard_cache(...)` for runtime cache collection
- `CacheSpec` for saying what a rule needs

## Public pieces

### `declare_blocks()`

Models can expose a semantic block decomposition with:

```python
def declare_blocks(self) -> list[BlockSpec]:
    ...
```

This is the noble path. Use it when a model has meaningful high-level blocks that a rule should target directly.

### `BlockSpec`

`BlockSpec` is the declarative description of a block.

Important fields:

- `name`
- `module`
- `rep`
- `is_output`
- `group`
- `params`
- `in_select`
- `out_select`

### `ResolvedBlock`

`ResolvedBlock` is the runtime block object returned in cache views.

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

### `CacheSpec`

`CacheSpec` tells the runtime what the rule needs.

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

## Views

The public views are:

- `declared`
- `execution`
- `paired_execution`

Important contract:

- `forward_with_standard_cache(...)` returns only the view requested by `CacheSpec.target_view`
- it does not return all three views anymore

Use:

- `declared` when the model exposes a semantic block decomposition
- `execution` for the universal auto-cache path
- `paired_execution` when a rule needs post-activation pairing

## Noble vs universal path

Noble path:

- model implements `declare_blocks()`
- rule can work from the model’s semantic block structure

Universal path:

- LeLabo hooks regular `nn.Module`s automatically
- this is more generic, but often lower-level

`paired_execution` is the universal path enriched with post-activation pairing when possible.

## Custom activation pairing

If your model uses a custom activation module, register it for cache pairing:

```python
from lelabo.models import register_cache_pair_activation
```

This helps `paired_execution` recover post-activation tensors without hardcoding your module into the runtime.

## `exec_module`

`ResolvedBlock.exec_module` is the best local replay module the runtime can expose for that block.

Typical use:

- local rules that need to replay a block or a small local segment

`exec_span_names` tells you which runtime names are covered by that replay module.

## Stable public contract vs runtime detail

Stable public contract:

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
- the exact content of `cache["_runtime"]`
- any internal reconstruction helpers used to build replay modules

## Current limits

What works well today:

- standard supervised runs
- many local-rule setups based on cached intermediate activations
- models that either declare blocks cleanly or expose usable `nn.Module` structure

What is not first-class yet:

- general staged multi-phase training programs
- native pretrain/freeze/fine-tune programs as public runtime objects
- arbitrary papers with multiple loaders, multiple phases, and multiple optimizers expressed as a first-class experiment program

Some papers can still be implemented, but the runtime does not yet present a dedicated staged-program abstraction.
