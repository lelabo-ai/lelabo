# Train Configs

This folder contains TOML examples for `lelabo train`.

Each config should declare:

- `config_version`: use `"auto"` (or explicit schema version, currently `1.0`)
- `lelabo_version`: use `"auto"` (resolved from installed LeLabo version)

Priority order at runtime:

1. Library defaults (`src/lab/config/assets/*.toml`, loaded by `src/lab/config/defaults.py`)
2. Project TOML (`train.<mode>.toml`, `train.toml`, `configs/train.<mode>.toml`, `configs/train.toml`)
3. Explicit CLI overrides
4. Advanced `--set key=value` overrides

`--config` is optional. If provided, it overrides the auto-detected project config path.

Examples:

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml --dataset iris --set early_stopping.patience=20
```

Built-in supervised metrics you can request in `[[metrics]]`:

- `accuracy` / `acc`
- `precision`, `recall`, `f1` (`average = macro|micro|weighted|binary`)
- `mse`, `mae`, `rmse`, `r2`

Example monitor:

```toml
[early_stopping]
monitor = "val.f1_macro"
```

For custom metric plugins inside a capsule, use `metrics/example.py` patterns:

1. Function metric with `register_metric_fn`
2. Advanced class metric with `ClassificationMetricBase` / `RegressionMetricBase`

```bash
lelabo train rl --config configs/train/rl.detailed.toml --env CartPole-v1
```
