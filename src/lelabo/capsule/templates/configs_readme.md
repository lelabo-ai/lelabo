# Configs

This folder is the recommended place for capsule-level experiment configuration.

## Supported workflow

Use layered config resolution:

1. Library defaults (embedded in `lelabo.config.defaults`)
2. Capsule/project config file (`train.toml` or explicit `--config`)
3. CLI scalar overrides (`--dataset`, `--lr`, etc.)
4. Advanced nested overrides (`--set key=value`)

`--config` is optional. If omitted, LeLabo auto-detects in this order:

1. `train.<mode>.toml` (example: `train.supervised.toml`)
2. `train.toml`
3. `configs/train.<mode>.toml`
4. `configs/train.toml`

## Quick commands

```bash
lelabo train supervised --config configs/train.supervised.quickstart.toml
```

```bash
lelabo train supervised --config configs/train.supervised.detailed.toml --dataset iris
```

```bash
lelabo train rl --config configs/train.rl.detailed.toml --env CartPole-v1
```

```bash
lelabo train supervised --set model.params.hidden=1024 --set scheduler.params.gamma=0.5
```

## Config schema style

Use `name + params` for extensible components:

- `dataset`
- `model`
- `initializer`
- `loss`
- `update_rule`
- `optimizer`
- `scheduler`
- `callbacks`
- `metrics` (optional list of built-ins + custom metric plugins)

Built-in supervised metrics include:

- `accuracy` / `acc`
- `precision`, `recall`, `f1` (supports `average = "macro" | "micro" | "weighted" | "binary"`)
- `mse`, `mae`, `rmse`, `r2`

For a custom metric plugin starter, see:

- `metrics/example.py`
  This file shows how to register a streaming metric with `register_metric`
  and `ClassificationStreamingMetric`.

For a custom initializer plugin starter, see:

- `initializers/example.py`
  This file shows how to register an initializer with `register_initializer`
  and explains the contract `builder(ctx) -> callable(model)`.

For a custom loss plugin starter, see:

- `losses/example.py`
  This file shows how to register a loss with `register_loss`
  and explains the contract `builder(ctx) -> callable(pred, target)`.

For a custom scheduler plugin starter, see:

- `schedulers/example.py`
  This file shows how to register a scheduler with `register_scheduler`.

For a custom optimizer plugin starter, see:

- `optimizers/example.py`
  This file shows how to register an optimizer with `register_optimizer`.

Quick custom optimizer snippet:

```toml
[optimizer]
name = "my_adamw"
[optimizer.params]
lr = 0.001
weight_decay = 0.01
betas = [0.9, 0.999]
eps = 1e-8
```

Quick custom initializer snippet:

```toml
[initializer]
name = "example_all_ones"
[initializer.params]
weight_value = 1.0
bias_value = 0.0
```

Quick custom loss snippet:

```toml
[loss]
name = "example_scaled_l1"
[loss.params]
scale = 0.5
```

Quick custom scheduler snippet:

```toml
[scheduler]
name = "my_cosine"
interval = "epoch"
monitor = "val.loss"
[scheduler.params]
T_max = 50
eta_min = 0.0
```

Quick built-in early stopping callback snippet:

```toml
[[callbacks]]
name = "earlystopping"
enabled = true
[callbacks.params]
monitor = "val.f1_macro"
mode = "auto"
patience = 5
min_delta = 0.0
warmup = 5
restore_best = true
```

For a custom callback plugin starter, see:

- `callbacks/example.py`
  This file shows how to register callbacks with `register_callback`.

Quick custom metric snippets:

```toml
[[metrics]]
name = "example_error_rate"
```

```toml
[[metrics]]
name = "example_error_rate"
[metrics.params]
key = "example_error_rate"
ignore_label = -100
min_samples = 16
as_percent = true
```

Example:

```toml
[[metrics]]
name = "f1"
[metrics.params]
average = "macro"
```

Then you can monitor directly:

```toml
[[callbacks]]
name = "earlystopping"
[callbacks.params]
monitor = "val.f1_macro"
```

Top-level version fields are recommended in every file:

- `config_version = "auto"` (or explicit schema version)
- `lelabo_version = "auto"` (resolved from installed LeLabo version)

Example:

```toml
[model]
name = "mlp"
[model.params]
hidden = 512
layers = 4
```

## Suggested naming convention

- `train.supervised.quickstart.toml` for simplest baseline
- `train.supervised.detailed.toml` for full research config
- `train.rl.detailed.toml` for RL config
- `train.supervised.<experiment>.toml` for custom runs

## Reproducibility tips

- Always set `runtime.seed`.
- Save run outputs with `runtime.run_dir`.
- Keep one config file per experiment family and avoid ad-hoc CLI-only runs.
- Prefer `--set` only for quick iteration; commit stable values to TOML files.

## Common troubleshooting

- `--dataset cannot be empty`: set `dataset.name` in config or pass `--dataset`.
- Unknown key/type in `params`: verify plugin expected fields.
- Scheduler errors: ensure scheduler `name` and `scheduler.params` match torch expectations.
