# Param Flow Reference

This file explains where custom config params arrive at runtime.

Use it when adding or debugging a capsule extension.

## Cheat sheet

- `[model.params] -> args.model_params`
- `[optimizer.params] -> ctx.optimizer_params()`
- `[scheduler.params] -> ctx.scheduler_params()`
- `[loss.params] -> ctx.loss_params()`
- `[initializer.params] -> ctx.initializer_params()`
- `[callbacks.params] -> ctx.callback_params()`
- `[metrics.params.<metric_name>] -> ctx.metric_params(name)`
- `[update_rule.params] -> ctx.extra["update_rule_params"]`
- `dataset.params -> dataset builder kwargs`

## Model params

TOML:

```toml
[model]
name = "example_mlp"

[model.params]
hidden = 128
layers = 3
dropout = 0.1
```

Read them in code with:

```python
params = dict(getattr(args, "model_params", {}) or {})
hidden = int(params.get("hidden", 128))
```

## Optimizer params

Normalized fields are already placed on the context:

- `ctx.lr`
- `ctx.weight_decay`
- `ctx.momentum`

Extra params come from:

```toml
[optimizer.params]
betas = [0.9, 0.999]
eps = 1e-8
```

Read them with:

```python
params = ctx.optimizer_params()
```

## Scheduler params

Normalized scheduling metadata is already on the context:

- `ctx.interval`
- `ctx.monitor`
- `ctx.epochs`
- `ctx.steps_per_epoch`
- `ctx.total_steps`

Extra params come from:

```toml
[scheduler.params]
T_max = 50
eta_min = 0.0
```

Read them with:

```python
params = ctx.scheduler_params()
```

## Loss params

TOML:

```toml
[loss.params]
scale = 0.5
```

Code:

```python
params = ctx.loss_params()
```

## Initializer params

TOML:

```toml
[initializer.params]
distribution = "uniform"
seed = 1234
```

Code:

```python
params = ctx.initializer_params()
```

## Callback params

TOML:

```toml
[[callbacks]]
name = "epoch_echo"

[callbacks.params]
every_n_epochs = 2
key = "val.loss"
```

Code:

```python
params = ctx.callback_params()
every_n_epochs = int(params.get("every_n_epochs", 1))
```

## Metric params

Metric params are namespaced by metric name.

TOML:

```toml
[[metrics]]
name = "example_error_rate"

[metrics.params.example_error_rate]
as_percent = true
```

Code:

```python
params = ctx.metric_params("example_error_rate")
```

## Update rule params

TOML:

```toml
[update_rule]
name = "local_head"

[update_rule.params]
average_grads = false
grad_clip = 1.0
```

Current runtime contract:

- primary source: `ctx.extra["update_rule_params"]`
- often mirrored on: `args.update_rule_params`

Typical code:

```python
from collections.abc import Mapping

params = {}
if isinstance(ctx.extra, Mapping):
    node = ctx.extra.get("update_rule_params", {})
    if isinstance(node, Mapping):
        params.update(dict(node))
from_args = getattr(ctx.args, "update_rule_params", None)
if isinstance(from_args, Mapping):
    params.update(dict(from_args))
```

## Dataset params

Dataset builders are different: they receive keyword args directly.

TOML:

```toml
[dataset]
name = "example_mnist"

[dataset.params]
batch_size = 128
flatten = true
```

Code:

```python
def make_example_mnist(*, batch_size: int = 128, flatten: bool = True, **_: object):
    ...
```

## Rule of thumb

- `models`: read custom params from `args.model_params`
- `datasets`: read kwargs directly
- everything else: use the context helper method first

If you are unsure, check `resources/LELABO_REFERENCE.md`.
