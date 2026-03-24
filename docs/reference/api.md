# Public API

LeLabo is CLI-first. The Python API documented here covers the extension points and runtime surfaces that are intentionally public and stable.

Everything not listed here should be treated as research-grade internal code.

---

## Registry decorators

The primary way to extend LeLabo. Each decorator registers a named builder that the CLI can resolve.

```python
from lelabo.models.registry import register_model, ModelContext
from lelabo.update_rules.registry import register_update_rule, UpdateRuleContext
from lelabo.supervised.datasets.registry import register_dataset
from lelabo.optimizers import register_optimizer, OptimizerContext
from lelabo.losses import register_loss
from lelabo.metrics import register_metric
from lelabo.schedulers import register_scheduler
from lelabo.callbacks import register_callback
```

### `@register_model(name)`

```python
@register_model("my_model")
def build_my_model(ctx: ModelContext, args) -> nn.Module:
    ...
```

`ModelContext` fields: `in_dim`, `num_classes`, `input_shape`

Model params: `dict(getattr(args, "model_params", {}) or {})`

### `@register_update_rule(name)`

```python
@register_update_rule("my_rule")
def build_my_rule(ctx: UpdateRuleContext):
    ...
```

`UpdateRuleContext` fields:

| Field | Description |
|---|---|
| `ctx.model` | The `nn.Module` being trained |
| `ctx.loss_fn` | The configured loss function — call as `ctx.loss_fn(output, target)` |
| `ctx.device` | The device string (`"cpu"`, `"cuda"`, etc.) |
| `ctx.extra` | Dict of additional context, including rule params |

Rule params: `ctx.extra.get("update_rule_params", {})`

### `@register_dataset(name)`

```python
@register_dataset("my_dataset")
def make_my_dataset(**kwargs) -> DataBundle:
    ...
```

Dataset params arrive as `**kwargs`. Must return a `DataBundle`.

### `@register_optimizer(name)`

```python
@register_optimizer("my_optimizer")
def build_my_optimizer(ctx: OptimizerContext) -> torch.optim.Optimizer:
    ...
```

`OptimizerContext` fields: `param_groups`, `lr`, `weight_decay`, `momentum`

Extra params: `ctx.optimizer_params()`

---

## Cache contract

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

### `CacheSpec`

```python
CacheSpec(
    target_view: str,                          # "declared" | "execution" | "paired_execution"
    trainable_module_types: tuple = (),
    observed_module_types: tuple = (),
    observed_module_names: tuple = (),
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    require_single_call: bool = False,
    require_single_output_head: bool = False,
    require_input_ndim: int | None = None,
    require_output_ndim: int | None = None,
)
```

### `forward_with_standard_cache()`

```python
out, cache, views = forward_with_standard_cache(
    model: nn.Module,
    x: Tensor,
    cache_spec: CacheSpec,
) -> tuple[Tensor, dict, list[ResolvedBlock]]
```

Returns only the view requested by `cache_spec.target_view`.

### `BlockSpec`

```python
BlockSpec(
    name: str,
    module: nn.Module,
    rep: nn.Module | None = None,       # activation module
    is_output: bool = False,
    group: str | None = None,
    params: list | None = None,
    in_select: ... = None,
    out_select: ... = None,
)
```

### `ResolvedBlock`

Key fields available on each block in a view:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Block identifier |
| `module` | `nn.Module` | The block's module |
| `x` | `Tensor \| None` | Input tensor |
| `u` | `Tensor \| None` | Pre-activation tensor |
| `h` | `Tensor \| None` | Post-activation tensor |
| `is_trainable` | `bool` | Has trainable parameters |
| `is_output` | `bool` | Output block |
| `exec_module` | `nn.Module \| None` | Best local replay module |

### `register_cache_pair_activation()`

```python
register_cache_pair_activation(MyActivationClass)
```

Register a custom activation module for post-activation pairing in `paired_execution`.

---

## Trainer

```python
from lelabo.core.trainer import Trainer
```

```python
result: FitResult = Trainer(
    model,
    learner,
    loss,
    device="cpu",
    display_mode="compact",
    callbacks=None,
    logger=None,
    schedulers=None,
    metrics=None,
).fit(train_loader, val_loader, epochs=20)
```

---

## Output types

```python
from lelabo.core.train_types import (
    FitResult,
    EpochRecord,
    SplitSummary,
    BestSummary,
    FitRuntime,
    MonitorStatus,
    RestorationStatus,
)
```

All output types are frozen dataclasses with a `.to_dict()` method.

---

## Callback base class

```python
from lelabo.core.callbacks import Callback

class MyCallback(Callback):
    def on_train_start(self, trainer, state): ...
    def on_epoch_start(self, trainer, state): ...
    def on_batch_start(self, trainer, state): ...
    def on_batch_end(self, trainer, state, logs): ...
    def on_epoch_end(self, trainer, epoch_record, state): ...
    def on_eval_end(self, trainer, split_summary, state): ...
    def on_train_end(self, trainer, fit_result, state): ...
```

---

## DataBundle

```python
from lelabo.supervised.datasets.bundle import DataBundle

DataBundle(
    train: Dataset,
    val: Dataset | None = None,
    test: Dataset | None = None,
    num_classes: int = 0,
    in_dim: int = 0,
    input_shape: tuple = (),
)
```
