# API Overview

LeLabo is CLI-first, but a few Python surfaces are worth knowing when extending the framework.

## Stability

The public extension points below are intentional.
Everything else should still be treated as research-grade internal code.

## Registry entrypoints

Main pattern:

```python
from lelabo.models.registry import ModelContext, register_model
from lelabo.update_rules.registry import register_update_rule
from lelabo.datasets import register_dataset
```

The main public idea is:

- register a named builder
- let the CLI resolve it by name
- keep the implementation close to plain PyTorch

## Supervised runtime

The supervised stack is built from:

- a regular PyTorch `nn.Module`
- a loss
- an update rule
- optional streaming metrics
- the trainer runtime

The practical entrypoint remains:

```bash
lelabo train supervised ...
```

The Python runtime surface worth knowing is:

```python
from lelabo.core.trainer import Trainer
```

## Cache provider

Main imports:

```python
from lelabo.models.cache_provider import (
    CacheSpec,
    forward_with_standard_cache,
    declares_blocks,
    resolve_declared_blocks,
)
from lelabo.models.blocks import BlockSpec, ResolvedBlock
from lelabo.models import register_cache_pair_activation
```

Current public cache contract:

- models may expose `declare_blocks()`
- cache collection is requested through `CacheSpec`
- `forward_with_standard_cache(...)` returns:
  - `out`
  - `cache`
  - `views`
- `forward_with_standard_cache(...)` returns only the requested `target_view`

Public view names:

- `declared`
- `execution`
- `paired_execution`

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
        target_view="execution",
        trainable_module_types=(nn.Linear,),
        require_single_call=True,
        require_single_output_head=True,
    ),
)
```

## What this page does not promise

This page does not document:

- internal cache pairing heuristics
- internals of `cache["_runtime"]`
- all helper functions under built-in update-rule implementations

Those are not public stability commitments.
