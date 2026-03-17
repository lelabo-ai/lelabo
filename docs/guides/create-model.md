# Create a Model

Register a custom `nn.Module` so it can be selected by name from a config or the CLI.

## When to use this

Use a model extension when the architecture changes. If you only need different hyperparameters on an existing architecture, use `--set` or edit the config instead.

## Use `nn.Module`, not `torch.nn.functional`

!!! tip "Prefer `nn.Module` layers over `torch.nn.functional`"
    We strongly encourage using module-based layers (`nn.ReLU()`, `nn.Linear(...)`) wherever possible. The autocache hooks onto `nn.Module` objects — functional calls (`F.relu`, `F.linear`) have no module to hook onto and won't be captured. If you don't need autocache or local rules, `F.*` still works fine.

## Where to edit

Inside your capsule: `models/example.py`

## Builder shape

```python
from lelabo.models.registry import ModelContext, register_model
import torch.nn as nn


@register_model("my_model")
def build_my_model(ctx: ModelContext, args):
    params = dict(getattr(args, "model_params", {}) or {})
    hidden = params.get("hidden", 128)
    in_dim = ctx.in_dim
    num_classes = ctx.num_classes

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )
```

`ModelContext` carries shape metadata resolved from the dataset:

| Field | Description |
|---|---|
| `in_dim` | Flattened input dimension |
| `num_classes` | Number of output classes |
| `input_shape` | Raw input shape tuple |

Model params (hidden size, depth, etc.) come from `args.model_params`, not from `ModelContext`.

## Config snippet

```toml
[model]
name = "my_model"

[model.params]
hidden = 256
```

## Verify

```bash
lelabo list models
# my_model should appear

lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --model my_model
```

## Common mistakes

- Reading model params from `ModelContext` instead of `args.model_params`
- Forgetting to handle the case where `model_params` is `None` or empty
- Leaving the `@register_model` decorator commented out

## Model with local-rule support

If your model needs to support local learning rules, implement `declare_blocks()`:

```python
from lelabo.models.registry import ModelContext, register_model
from lelabo.models.blocks import BlockSpec
import torch.nn as nn


@register_model("my_local_model")
def build_my_local_model(ctx: ModelContext, args):
    params = dict(getattr(args, "model_params", {}) or {})

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(ctx.in_dim, 128)
            self.act1 = nn.ReLU()
            self.layer2 = nn.Linear(128, ctx.num_classes)

        def forward(self, x):
            x = self.act1(self.layer1(x))
            return self.layer2(x)

        def declare_blocks(self) -> list[BlockSpec]:
            return [
                BlockSpec(name="layer1", module=self.layer1, rep=self.act1),
                BlockSpec(name="layer2", module=self.layer2, is_output=True),
            ]

    return MyModel()
```

See [Cache & local rules](../concepts/cache-local-rules.md) for the full `BlockSpec` contract.
