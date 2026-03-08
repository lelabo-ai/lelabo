# API overview

LeLabo is primarily CLI-first, but a few Python entrypoints are worth knowing when extending the framework.

## Stability

The Python API is still research-grade.

- extension points are intentional
- internals can still move
- prefer registries and builders over deep imports when possible

## Registries

The main extension pattern is registry-based.

Model registration:

```python
from lelabo.models.registry import ModelContext, register_model
```

Typical builder shape:

```python
@register_model("my_model")
def build_my_model(ctx: ModelContext, args):
    ...
```

At runtime, the CLI resolves the builder by name and calls it with:

- a `ModelContext`
- parsed CLI/config arguments

Useful helper:

```python
from lelabo.models.registry import build_model
```

## Tasks

Supervised tasks live in `src/lelabo/core/task.py`.

Common entrypoints:

- `ClassificationTask`
- `GLUETask`

These objects define:

- `loss(...)`
- `metrics(...)`
- `output_deltas(...)` when needed by local rules

## Cache provider

The cache provider is the main bridge between plain `nn.Module` models and local learning rules.

Imports:

```python
from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache
```

`forward_with_standard_cache(...)` returns:

- `out`
- `cache`
- `views`

Important `views` keys:

- `execution_blocks`
- `output_blocks`
- `declared_blocks`
- `trainable_segments`

This is the contract used by built-in local rules such as DFA, DRTP, FA, DNI, SCL, and SoftHebb.

## Minimal cache example

```python
import torch
import torch.nn as nn

from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 5),
)
x = torch.randn(4, 10)

out, cache, views = forward_with_standard_cache(
    model,
    x,
    cache_spec=CacheSpec(
        trainable_module_types=(nn.Linear,),
        require_single_call=True,
        require_single_output_head=True,
    ),
)
```

## Update rules

Built-in update rules are registered by short names such as:

- `bp`
- `dfa`
- `drtp`
- `fa`
- `dni`
- `scl`
- `softhebb`

From the CLI:

```bash
lelabo list update-rules
```

For most users, the correct entrypoint is still the CLI rather than manually instantiating the trainer stack.
