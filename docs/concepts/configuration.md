# Configuration

LeLabo uses TOML config files for experiments. Every parameter has a default, and every default can be overridden — from the config file, from the CLI, or from `--set` for nested values.

## Resolution order

Configuration is resolved in four layers, each overriding the previous:

```
1. Library defaults
2. Config file  (--config path/to/config.toml)
3. CLI flags    (--dataset, --model, --lr, --epochs, …)
4. --set flags  (--set model.params.hidden=256)
```

The resolved config is always written to `resolved_config.yaml` in the run directory — so you always know exactly what ran.

## Config file structure

A supervised config has these top-level blocks:

```toml
config_version = "auto"
lelabo_version = "auto"
task = "supervised"

[dataset]
name = "iris"
[dataset.params]
# dataset-specific params

[model]
name = "mlp"
[model.params]
hidden = 128
layers = 2
activation = "relu"

[initializer]
name = "torch_default"
[initializer.params]

[loss]
name = "ce"
[loss.params]

[update_rule]
name = "bp"
[update_rule.params]

[optimizer]
name = "adamw"
[optimizer.params]
lr = 0.001
weight_decay = 0.0

[scheduler]
name = "none"
interval = "epoch"
monitor = "val.loss"
[scheduler.params]

[runtime]
device = "auto"
seed = 2
determinism = "relaxed"
display = "compact"
run_dir = ""           # set to enable artifact writing
save_checkpoints = false

[train]
epochs = 20
batch = 32
val_frac = 0.1

[[callbacks]]
name = "earlystopping"
enabled = true
[callbacks.params]
monitor = "val.acc"
patience = 5
restore_best = true

[[metrics]]
name = "acc"
```

## Annotated example

Here is a real config file — `configs/train/supervised.quickstart.toml` — with every section explained.

```toml
config_version = "auto"                # (1)!
lelabo_version = "auto"                # (2)!
task = "supervised"                    # (3)!

[dataset]
name = "iris"                          # (4)!
[dataset.params]                       # (5)!

[model]
name = "mlp"                           # (6)!
[model.params]
hidden = 128                           # (7)!
layers = 2
activation = "relu"

[initializer]
name = "torch_default"                 # (8)!
[initializer.params]

[loss]
name = "ce"                            # (9)!
[loss.params]

[update_rule]
name = "bp"                            # (10)!
[update_rule.params]

[optimizer]
name = "adamw"                         # (11)!
[optimizer.params]
lr = 0.001                             # (12)!
weight_decay = 0.0

[scheduler]
name = "none"                          # (13)!
interval = "epoch"
monitor = "val.loss"
[scheduler.params]

[runtime]
device = "auto"                        # (14)!
seed = 2                               # (15)!
determinism = "relaxed"                # (16)!
display = "compact"

[train]
epochs = 20                            # (17)!
batch = 32
val_frac = 0.1                         # (18)!

[[callbacks]]                          # (19)!
name = "earlystopping"
enabled = true
[callbacks.params]
monitor = "val.acc"                    # (20)!
patience = 5
restore_best = true

[[metrics]]                            # (21)!
name = "acc"
```

1. Config format version — leave as `"auto"`, LeLabo handles it.
2. LeLabo version constraint — `"auto"` means no constraint.
3. Task type. `"supervised"` or `"rl"`.
4. Dataset name — must match a registered dataset. See `lelabo list datasets`.
5. Dataset-specific params passed as `**kwargs` to the dataset builder. Iris has none.
6. Model name — must match a registered model. See `lelabo list models`.
7. Model params — passed to the model builder as `args.model_params`. Each model defines its own params.
8. Weight initialization strategy. `"torch_default"` uses PyTorch defaults.
9. Loss function. `"ce"` = cross-entropy, `"mse"` = mean squared error, `"bce"` = binary cross-entropy.
10. Learning rule. `"bp"` = standard backpropagation. See `lelabo list update-rules` for alternatives.
11. Optimizer name. Standard choices: `"sgd"`, `"adam"`, `"adamw"`.
12. Optimizer params — `lr` and `weight_decay` are common. Additional params are available via `ctx.optimizer_params()` in the builder.
13. LR scheduler. `"none"` disables scheduling. Options: `"step"`, `"exponential"`, `"cosine"`, `"reduce_on_plateau"`.
14. Device selection. `"auto"` picks CUDA if available, falls back to CPU.
15. Random seed for reproducibility.
16. Determinism level. `"relaxed"` = seed set but non-deterministic CUDA ops allowed. `"strict"` = fully deterministic, slower. `"off"` = no seed.
17. Training epochs and batch size.
18. If the dataset has no validation split, this fraction of training data is held out.
19. Callbacks use TOML array syntax `[[callbacks]]` — you can have multiple.
20. Which scalar to monitor for early stopping. Format: `{split}.{metric}`.
21. Metrics also use array syntax. Each entry adds a metric computed every epoch.

## Component blocks

Each extensible component follows the same pattern:

```toml
[component]
name = "registered_name"

[component.params]
param1 = value1
param2 = value2
```

The `name` selects a registered builder. The `params` block is passed to that builder. This pattern is the same for `dataset`, `model`, `loss`, `update_rule`, `optimizer`, `scheduler`, `callbacks`, and `metrics`.

## CLI overrides

Common fields have dedicated CLI flags:

```bash
lelabo train supervised \
  --dataset cifar10 \
  --model cnn \
  --rule dfa \
  --optimizer adam \
  --lr 0.0003 \
  --epochs 100 \
  --batch 64
```

## `--set` for nested overrides

For anything without a dedicated flag, use `--set` with a dotted path:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --set model.params.hidden=512 \
  --set model.params.layers=4 \
  --set scheduler.params.gamma=0.95
```

`--set` is repeatable and always takes priority over the config file.

## Runtime options

| Key | Default | Description |
|---|---|---|
| `runtime.device` | `"auto"` | `"cpu"`, `"cuda"`, `"mps"`, or `"auto"` |
| `runtime.seed` | `2` | Random seed |
| `runtime.determinism` | `"relaxed"` | `"off"`, `"relaxed"`, or `"strict"` |
| `runtime.display` | `"compact"` | Console output: `"none"`, `"compact"`, or `"rich"` |
| `runtime.run_dir` | `""` | Path to write run artifacts (empty = disabled) |
| `runtime.save_checkpoints` | `false` | Write `checkpoints/last.pt` and `checkpoints/best.pt` |

## W&B configuration

Weights & Biases logging is configured via the `[wandb]` section:

```toml
[wandb]
project = "my-research"
entity = "my-team"
tags = ["dfa", "baseline"]
group = "experiment-v1"
```

Environment variables `WANDB_PROJECT` and `WANDB_ENTITY` serve as fallbacks. Config file values take precedence over environment variables. See the [W&B guide](../guides/wandb.md) and the [config schema reference](../reference/config-schema.md#wandb) for all options.

## Auto-detection

If you don't pass `--config`, LeLabo looks for a config file automatically in the current directory:

```
train.supervised.toml
train.toml
configs/train/supervised.toml
configs/train.toml
```

The first file found is used. This is convenient when working inside a capsule that already has its own config.

## Config in capsules

Capsules come with their own config files in `configs/`. These are the entry points for capsule-specific experiments:

```
configs/
├── train/supervised.quickstart.toml
├── train/supervised.capsule_optimizer.toml
├── train/supervised.detailed.toml
└── train/supervised.paper_pack.toml
```

The capsule configs reference the same component names as the global registry — plus any names registered by the capsule itself.
