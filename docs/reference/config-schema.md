# Config Schema

Complete reference for every key in a supervised TOML config file.

## Top-level

| Key | Type | Default | Description |
|---|---|---|---|
| `config_version` | str | `"auto"` | Config format version |
| `lelabo_version` | str | `"auto"` | LeLabo version constraint |
| `task` | str | — | `"supervised"` or `"rl"` |

---

## `[dataset]`

| Key | Type | Description |
|---|---|---|
| `name` | str | Registered dataset name |
| `params` | table | Passed as `**kwargs` to the dataset builder |

---

## `[model]`

| Key | Type | Description |
|---|---|---|
| `name` | str | Registered model name |
| `params` | table | Available as `args.model_params` in the builder |

Built-in models: `mlp`, `cnn`, `resnet18`, `bert`, `deephebb`

---

## `[initializer]`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"torch_default"` | Registered initializer name |
| `params` | table | — | Initializer-specific params |

---

## `[loss]`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"ce"` | Registered loss name |
| `params` | table | — | Loss-specific params |

Built-in losses: `ce` (cross-entropy), `mse`, `bce`

---

## `[update_rule]`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"bp"` | Registered update rule name |
| `params` | table | — | Available as `ctx.extra["update_rule_params"]` |

Built-in rules: `bp`, `dfa`, `drtp`, `fa`, `dni`, `scl`, `softhebb`

---

## `[optimizer]`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"adamw"` | Registered optimizer name |
| `params.lr` | float | `0.001` | Learning rate |
| `params.weight_decay` | float | `0.0` | Weight decay |
| `params.*` | — | — | Additional params passed to `ctx.optimizer_params()` |

---

## `[scheduler]`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"none"` | Registered scheduler name |
| `interval` | str | `"epoch"` | Step interval: `"epoch"` or `"step"` |
| `monitor` | str | `"val.loss"` | Scalar to monitor for adaptive schedulers |
| `params` | table | — | Scheduler-specific params |

---

## `[runtime]`

| Key | Type | Default | Description |
|---|---|---|---|
| `device` | str | `"auto"` | `"cpu"`, `"cuda"`, `"mps"`, or `"auto"` |
| `seed` | int | `2` | Random seed |
| `determinism` | str | `"relaxed"` | `"off"`, `"relaxed"`, or `"strict"` |
| `display` | str | `"compact"` | Console output: `"none"`, `"compact"`, `"rich"` |
| `run_dir` | str | `""` | Path to write run artifacts. Empty = disabled. |
| `save_checkpoints` | bool | `false` | Write `checkpoints/last.pt` and `checkpoints/best.pt` |

---

## `[train]`

| Key | Type | Default | Description |
|---|---|---|---|
| `epochs` | int | `20` | Number of training epochs |
| `batch` | int | `32` | Batch size |
| `val_frac` | float | `0.1` | Fraction of training data to use for validation (if no val split) |
| `input_noise_dataset` | float | `0.0` | Gaussian noise std applied to training inputs (for robustness experiments) |
| `noise_on_test` | bool | `false` | Also apply input noise during test evaluation |

---

## `[[callbacks]]`

Array of callback entries. Each entry:

| Key | Type | Description |
|---|---|---|
| `name` | str | Registered callback name |
| `enabled` | bool | Whether this callback is active |
| `params` | table | Callback-specific params |

### `earlystopping` params

| Key | Type | Default | Description |
|---|---|---|---|
| `monitor` | str | `"val.acc"` | Scalar to monitor, e.g. `val.acc`, `val.loss` |
| `mode` | str | `"auto"` | `"auto"`, `"max"`, or `"min"` |
| `patience` | int | `5` | Epochs without improvement before stopping |
| `min_delta` | float | `0.0` | Minimum change to count as improvement |
| `warmup` | int | `3` | Epochs before monitoring starts |
| `restore_best` | bool | `true` | Restore best model state at the end of training |

---

## `[[metrics]]`

Array of metric entries.

| Key | Type | Description |
|---|---|---|
| `name` | str | Registered metric name |
| `params` | table | Metric-specific params |

Built-in metrics: `acc`, `f1`, `precision`, `recall`, `mse`, `mae`, `r2`

---

## `[wandb]`

Optional [Weights & Biases](https://wandb.ai) integration. See the [W&B guide](../guides/wandb.md) for setup and usage.

| Key | Type | Default | Description |
|---|---|---|---|
| `project` | str | `None` | W&B project name. **Required to enable logging.** Falls back to `WANDB_PROJECT` env var. |
| `entity` | str | `None` | W&B team or user. Falls back to `WANDB_ENTITY` env var. |
| `tags` | list | `[]` | Tags for filtering runs in the W&B dashboard |
| `group` | str | `None` | Group name for organizing related runs (auto-set by sweeps) |
| `notes` | str | `""` | Free-text notes attached to the run |
| `enabled` | bool | `true` | Set to `false` to disable W&B even when project is set |

Example:

```toml
[wandb]
project = "local-learning-rules"
entity = "my-team"
tags = ["dfa", "mnist"]
group = "experiment-v2"
notes = "Comparing DFA variants on MNIST"
enabled = true
```

!!! note "W&B is optional"
    If the `wandb` package is not installed or `project` is not set, training proceeds normally without logging to W&B.

---

## `[robustness]`

Optional robustness evaluation settings. Used for repeated evaluation under noise or perturbation.

!!! warning "Experimental"
    This section is not yet fully documented. The fields exist in the schema but robustness evaluation workflows are not yet covered by guides.

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | str | `"none"` | Robustness evaluation mode. `"none"` disables it. |
| `trials` | int | `30` | Number of evaluation trials |
| `max_samples` | int | `0` | Max samples per trial (0 = no limit) |
