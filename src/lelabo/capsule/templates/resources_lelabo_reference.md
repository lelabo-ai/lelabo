# LeLabo Capsule Reference

This file is the local contract reference for capsule extensions.

Use it when you need the exact builder signature, return type, or context fields.

## Discovery rules

- LeLabo auto-loads Python files under:
  - `models/`
  - `update_rules/`
  - `datasets/`
  - `metrics/`
  - `initializers/`
  - `losses/`
  - `optimizers/`
  - `schedulers/`
  - `callbacks/`
- A plugin appears in `lelabo list ...` only after its `@register_*` decorator is uncommented.
- If the file imports fail, the plugin will not load.
- Helper/reference files are allowed, but only registered plugins are exposed.

## Builder signatures

| Folder | Register with | Builder signature | Must return |
| --- | --- | --- | --- |
| `models/` | `@register_model(name)` | `build_xxx(ctx, args)` | `nn.Module` |
| `update_rules/` | `@register_update_rule(name)` | `build_xxx(ctx)` | `UpdateRule` |
| `datasets/` | `@register_dataset(name)` | `build_xxx(**kwargs)` | `DataBundle` |
| `metrics/` | `@register_metric(name, kind=...)` | `build_xxx(ctx)` | `TrainerMetric` |
| `initializers/` | `@register_initializer(name)` | `build_xxx(ctx)` | `callable(model)` |
| `losses/` | `@register_loss(name)` | `build_xxx(ctx)` | `callable(pred, target)` |
| `optimizers/` | `@register_optimizer(name)` | `build_xxx(ctx)` | `torch.optim.Optimizer` |
| `schedulers/` | `@register_scheduler(name)` | `build_xxx(ctx)` | scheduler or `SchedulerController` |
| `callbacks/` | `@register_callback(name)` | `build_xxx(ctx)` | callback object |

## Contexts

### `ModelContext`

Fields:

- `dataset: str`
- `num_classes: int`
- `in_dim: int | None`
- `in_channels: int | None`
- `input_shape: tuple[int, ...] | None`
- `extra: dict[str, Any] | None`

Use it for dataset-shape metadata. For model-specific params, read `args.model_params`.

### `OptimizerContext`

Fields:

- `params`
- `lr: float`
- `weight_decay: float`
- `momentum: float`
- `args`
- `mode`
- `dataset`
- `params_extra`
- `extra`

Helpers:

- `ctx.optimizer_params() -> dict[str, Any]`

### `SchedulerContext`

Fields:

- `optimizer`
- `args`
- `epochs: int | None`
- `steps_per_epoch: int | None`
- `interval: str`
- `monitor: str`
- `params`
- `extra`

Helpers:

- `ctx.scheduler_params() -> dict[str, Any]`
- `ctx.total_steps -> int | None`

### `CallbackContext`

Fields:

- `args`
- `mode`
- `dataset`
- `model`
- `optimizer`
- `schedulers`
- `params`
- `index`
- `extra`

Helpers:

- `ctx.callback_params()`
- `ctx.callback_param(key, default)`
- `ctx.namespaced_params(*aliases)`

### `MetricContext`

Fields:

- `args`
- `mode`
- `dataset`
- `algo`
- `extra`

Helpers:

- `ctx.metric_params(name, *aliases)`

### `LossContext`

Fields:

- `args`
- `mode`
- `dataset`
- `task`
- `num_classes`
- `params`
- `extra`

Helpers:

- `ctx.loss_params()`

### `InitializerContext`

Fields:

- `args`
- `mode`
- `dataset`
- `model_name`
- `params`
- `extra`

Helpers:

- `ctx.initializer_params()`

### `UpdateRuleContext`

Fields:

- `args`
- `model`
- `task`
- `optimizer`
- `mode`
- `dataset`
- `rl_algo`
- `extra`

Notes:

- In the supervised runtime today, `ctx.task` is `None`.
- Update-rule-specific params usually arrive through `ctx.extra["update_rule_params"]`
  and are often mirrored on `args.update_rule_params`.

## `DataBundle`

Dataset builders must return a `DataBundle` with these fields:

- `train_loader`
- `val_loader`
- `test_loader`
- `num_classes: Optional[int]`
- `in_dim: Optional[int]`
- `input_shape: Optional[tuple[int, ...]]`
- `x_test: Optional[torch.Tensor]`
- `y_test: Optional[torch.Tensor]`
- `test_dataset: Optional[Dataset]`
- `meta: dict[str, Any]`

At minimum, define the loaders and the metadata your model path needs.

## Metric contract

Trainer metrics should use the streaming metric API from `lelabo.metrics`.

Important rules:

- `compute()` and `finalize()` must return numeric scalars only
- classification vs regression compatibility is validated by LeLabo
- metric params are namespaced by metric name

## Common real causes of failure

- Plugin missing from `lelabo list`
  - decorator still commented
  - import error in the plugin file
  - wrong folder

- `TypeError` from builder
  - wrong return type for the extension kind

- Model plugin cannot read params
  - model-specific params live on `args.model_params`, not on `ModelContext`

- Callback or metric params missing
  - read them through the helper methods on the context, not from raw nested config guesses

- Local rule breaks with a model
  - the model likely does not expose the expected cache/block structure
  - read `resources/MODEL_CACHE_ADVANCED.md`
