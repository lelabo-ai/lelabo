# Cache & Local Rules

Local learning rules — DFA, DNI, SoftHebb, and similar — need access to intermediate activations during the forward pass. LeLabo provides a stable public contract for this without requiring a custom training loop or a framework-specific model base class.

## The problem

A standard backprop training loop gives you gradients. A local rule needs something different: the pre-activation and post-activation tensors at each layer, the input to each block, or a specific pairing of intermediate states.

The naive solution is to write a custom training script for each rule. LeLabo avoids that by separating the cache collection contract from the update rule implementation.

## The public contract

Three pieces make up the cache contract:

### `declare_blocks()`

A model can expose a semantic block decomposition by implementing:

```python
def declare_blocks(self) -> list[BlockSpec]:
    ...
```

This is the clean path. Use it when your model has meaningful high-level blocks that a rule should target directly — for example, the layers of an MLP or the stages of a ResNet.

### `CacheSpec`

`CacheSpec` tells the runtime what the update rule needs from the forward pass:

```python
from lelabo.models.cache_provider import CacheSpec

cache_spec = CacheSpec(
    target_view="execution",           # which view to return
    trainable_module_types=(nn.Linear,),
    capture_inputs=True,
    capture_outputs=True,
    require_single_call=True,
    require_single_output_head=True,
)
```

Key fields:

| Field | Description |
|---|---|
| `target_view` | Which view to return: `"declared"`, `"execution"`, or `"paired_execution"` |
| `trainable_module_types` | Module types to treat as trainable blocks |
| `observed_module_types` | Additional module types to observe |
| `capture_inputs` | Whether to capture block inputs |
| `capture_outputs` | Whether to capture block outputs |
| `require_single_call` | Assert the model is called exactly once |
| `require_single_output_head` | Assert a single output tensor |

### `forward_with_standard_cache()`

Runs the forward pass and collects the cache in one call:

```python
from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache

out, cache, views = forward_with_standard_cache(
    model, x, cache_spec=cache_spec
)
```

Returns:

- `out` — the model output
- `cache` — raw cache dict
- `views` — the requested view (a list of `ResolvedBlock` objects)

## Views

The view is what the update rule actually works with. Three views are available:

| View | When to use |
|---|---|
| `declared` | The model implements `declare_blocks()` and exposes a semantic block structure |
| `execution` | Universal path — LeLabo hooks regular `nn.Module`s automatically |
| `paired_execution` | Universal path enriched with post-activation pairing when possible |

`forward_with_standard_cache()` returns only the view requested by `CacheSpec.target_view`.

## `ResolvedBlock`

Each entry in a view is a `ResolvedBlock`:

| Field | Description |
|---|---|
| `name` | Block identifier |
| `module` | The `nn.Module` for this block |
| `x` | Input tensor (if captured) |
| `u` | Pre-activation tensor (if available) |
| `h` | Post-activation tensor (if available) |
| `is_trainable` | Whether this block has trainable parameters |
| `is_output` | Whether this is the output block |
| `exec_module` | Best local replay module the runtime can expose |

## Noble path vs universal path

**Noble path** — the model implements `declare_blocks()`. The rule works from the model's own semantic block structure. Precise, explicit, recommended for new model implementations.

**Universal path** — the model is a plain `nn.Module`. LeLabo instruments it automatically. More generic, but the block boundaries are inferred rather than declared.

`paired_execution` is the universal path extended with post-activation pairing, useful when a rule needs both pre- and post-activation tensors.

## Example

```python
import torch.nn as nn
from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 5),
)

out, cache, views = forward_with_standard_cache(
    model,
    x,
    cache_spec=CacheSpec(
        target_view="execution",
        trainable_module_types=(nn.Linear,),
        require_single_call=True,
        require_single_output_head=True,
    ),
)

for block in views:
    print(block.name, block.x.shape, block.h.shape)
```

## Custom activation pairing

If your model uses a custom activation module, register it so `paired_execution` can recover post-activation tensors:

```python
from lelabo.models import register_cache_pair_activation

register_cache_pair_activation(MyCustomActivation)
```

## What is stable

The following are stable public contracts:

- `declare_blocks()` / `BlockSpec`
- `ResolvedBlock`
- `CacheSpec`
- `forward_with_standard_cache()`
- Views: `declared`, `execution`, `paired_execution`
- `register_cache_pair_activation()`

The following are runtime internals, not stable:

- The exact pairing heuristic
- The contents of `cache["_runtime"]`
- Internal reconstruction helpers for replay modules

## Current limits

What works well today:

- Standard supervised runs with local rules
- Models that implement `declare_blocks()` or expose usable `nn.Module` structure
- Most setups based on cached intermediate activations

What is not first-class yet:

- Staged multi-phase training programs
- Papers with multiple loaders, multiple optimizers, or explicit phase transitions

See [Research Notes](../research-notes.md) for more detail on current limitations.
