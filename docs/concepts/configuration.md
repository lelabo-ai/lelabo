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

## Auto-detection

If you don't pass `--config`, LeLabo looks for a config file automatically in the current directory:

```
train.supervised.toml
train.toml
configs/train.supervised.toml
configs/train.toml
```

The first file found is used. This is convenient when working inside a capsule that already has its own config.

## Config in capsules

Capsules come with their own config files in `configs/`. These are the entry points for capsule-specific experiments:

```
configs/
├── train.supervised.quickstart.toml
├── train.supervised.capsule_optimizer.toml
├── train.supervised.detailed.toml
└── train.supervised.paper_pack.toml
```

The capsule configs reference the same component names as the global registry — plus any names registered by the capsule itself.
