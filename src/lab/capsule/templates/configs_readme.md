# Configs

This folder is the recommended place for capsule-level experiment configuration.

## Supported workflow

Use layered config resolution:

1. Library defaults (embedded in `lab.config.defaults`)
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
lelabo train supervised --set model.params.hidden=1024 --set early_stopping.patience=20
```

## Config schema style

Use `name + params` for extensible components:

- `dataset`
- `model`
- `update_rule`
- `optimizer`
- `scheduler`
- `early_stopping`
- `metrics` (list)

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
