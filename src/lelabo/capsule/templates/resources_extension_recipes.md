# Extension Recipes

Use this file for the procedural "how do I add X?" path.
Use these recipes for normal capsule work. Open the reference docs only to
confirm a contract, not as the default starting point.

Open `AGENTS.md` first if you still need to choose the right extension type.
Open `resources/LELABO_REFERENCE.md` only when you need an exact signature or
context contract.

## Choose the right extension type first

- Architecture change -> model
- Learning or update logic -> update rule
- Data loading or split logic -> dataset
- Tuning only -> config
- Several moving parts -> paper pack

## Add an optimizer

### When to use this extension type

Use an optimizer plugin when you need a new optimizer builder, not just new
optimizer params.

### Where to edit

- Code: `optimizers/example.py`
- Config: `configs/train/supervised.capsule_optimizer.toml`
- Validation: `tests/test_capsule_optimizer_smoke.py`

### Builder shape

```python
from lelabo.optimizers import OptimizerContext, register_optimizer


# @register_optimizer("my_optimizer")
def build_my_optimizer(ctx: OptimizerContext):
    ...
```

### Where params come from

- normalized fields: `ctx.lr`, `ctx.weight_decay`, `ctx.momentum`
- extra params: `ctx.optimizer_params()`

### Minimal config snippet

```toml
[optimizer]
name = "my_optimizer"

[optimizer.params]
lr = 0.01
```

### Validation commands

```bash
lelabo list optimizers
```

```bash
lelabo train supervised --config configs/train/supervised.capsule_optimizer.toml --optimizer my_optimizer
```

```bash
pytest -q tests/test_capsule_optimizer_smoke.py
```

### Common mistakes

- leaving the decorator commented
- returning something that is not a real `torch.optim.Optimizer`
- reading params from the wrong place instead of `ctx.optimizer_params()`

### Core done criteria

- appears in `lelabo list optimizers`
- config runs
- params are read from the documented source

### Extra done criteria

- smoke test passes if the scaffold provides one
- runtime assumptions are documented if fit is partial

## Add a BP model

### When to use this extension type

Use a model plugin when the network architecture changes and normal BP is still
enough.

### Where to edit

- Code: `models/example.py`
- Config: `configs/train/supervised.quickstart.toml`

### Builder shape

```python
from lelabo.models.registry import ModelContext, register_model


# @register_model("my_model")
def build_my_model(ctx: ModelContext, args):
    params = dict(getattr(args, "model_params", {}) or {})
    ...
```

### Where params come from

- model params: `args.model_params`
- shape metadata: `ModelContext`

### Minimal config snippet

```toml
[model]
name = "my_model"

[model.params]
hidden = 128
```

### Validation commands

```bash
lelabo list models
```

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml --model my_model
```

### Common mistakes

- reading model params from `ModelContext` instead of `args.model_params`
- expecting cache features for a normal BP-only model
- forgetting to document extra assumptions about input shape

### Core done criteria

- appears in `lelabo list models`
- config runs
- params are read from the documented source

### Extra done criteria

- smoke test passes if the scaffold provides one
- runtime assumptions are documented if fit is partial

## Add a local-rule update rule

### When to use this extension type

Use an update rule when the learning dynamics change. If the rule needs
intermediate activations, also read `resources/MODEL_CACHE_ADVANCED.md`.

### Where to edit

- Code: `update_rules/example.py`
- Config: `configs/train/supervised.paper_pack.toml`
- Validation: `tests/test_paper_pack_smoke.py`

### Builder shape

```python
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule


# @register_update_rule("my_rule")
def build_my_rule(ctx: UpdateRuleContext):
    ...
```

### Where params come from

- primary source: `ctx.extra["update_rule_params"]`
- optional mirror: `args.update_rule_params`

### Minimal config snippet

```toml
[update_rule]
name = "my_rule"

[update_rule.params]
grad_clip = 1.0
```

### Validation commands

```bash
lelabo list update-rules
```

```bash
lelabo train supervised --config configs/train/supervised.paper_pack.toml --rule my_rule
```

```bash
pytest -q tests/test_paper_pack_smoke.py
```

### Common mistakes

- returning tensors or nested objects instead of numeric scalars
- assuming several cache views are available at once
- depending on block structure the model does not expose

### Core done criteria

- appears in `lelabo list update-rules`
- config runs
- params are read from the documented source

### Extra done criteria

- smoke test passes if the scaffold provides one
- runtime assumptions are documented if fit is partial

## Add a dataset

### When to use this extension type

Use a dataset plugin when you need custom loading, splitting, or metadata.

### Where to edit

- Code: `datasets/example.py`
- Config: `configs/train/supervised.quickstart.toml`

### Builder shape

```python
from lelabo.supervised.datasets.registry import register_dataset


# @register_dataset("my_dataset")
def make_my_dataset(**kwargs):
    ...
```

### Where params come from

- dataset params arrive as keyword args
- builders must return a `DataBundle`

### Minimal config snippet

```toml
[dataset]
name = "my_dataset"

[dataset.params]
batch_size = 128
```

### Validation commands

```bash
lelabo list datasets
```

```bash
lelabo train supervised --dataset my_dataset --model mlp
```

### Common mistakes

- returning something other than `DataBundle`
- hiding important metadata instead of setting `num_classes`, `in_dim`, or `input_shape`
- trying to read dataset params from a context helper

### Core done criteria

- appears in `lelabo list datasets`
- config runs
- params are read from the documented source

### Extra done criteria

- smoke test passes if the scaffold provides one
- runtime assumptions are documented if fit is partial

## Add a paper pack

### When to use this extension type

Use a paper pack when the idea spans several components such as model + update
rule + dataset + config + test.

### Where to edit

- Workflow: `resources/PAPER_PACK_PLAYBOOK.md`
- Config: `configs/train/supervised.paper_pack.toml`
- Validation: `tests/test_paper_pack_smoke.py`

### Shape

Decompose the paper into:

- model
- update rule
- dataset if needed
- config
- smoke test

### Where params come from

Use the documented source for each component:

- model -> `args.model_params`
- update rule -> `ctx.extra["update_rule_params"]`
- dataset -> kwargs
- optimizer/loss/callback/scheduler -> context helper methods

### Minimal config snippet

```toml
[model]
name = "my_model"

[update_rule]
name = "my_rule"
```

### Validation commands

```bash
lelabo train supervised --config configs/train/supervised.paper_pack.toml
```

```bash
pytest -q tests/test_paper_pack_smoke.py
```

### Common mistakes

- starting from one opaque script instead of decomposing the paper
- adding custom code where built-ins plus config would be enough
- hiding runtime limitations instead of documenting them

### Core done criteria

- config runs
- params are read from the documented source

### Extra done criteria

- smoke test passes if the scaffold provides one
- runtime assumptions are documented if fit is partial

## Short appendix: secondary extension types

Use the same pattern as above, but keep the change as small as possible.

### Metrics

- edit `metrics/example.py`
- register with `@register_metric(name, kind=...)`
- read params with `ctx.metric_params(name)`

### Losses

- edit `losses/example.py`
- register with `@register_loss(name)`
- read params with `ctx.loss_params()`

### Initializers

- edit `initializers/example.py`
- register with `@register_initializer(name)`
- read params with `ctx.initializer_params()`

### Callbacks

- edit `callbacks/example.py`
- register with `@register_callback(name)`
- read params with `ctx.callback_params()`

### Schedulers

- edit `schedulers/example.py`
- register with `@register_scheduler(name)`
- use `ctx.interval`, `ctx.monitor`, `ctx.total_steps`, and `ctx.scheduler_params()`

For exact contracts, read `resources/LELABO_REFERENCE.md`.
For param mapping, read `resources/PARAM_FLOW.md`.
