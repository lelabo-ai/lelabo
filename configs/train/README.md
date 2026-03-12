# Train Configs

This folder contains TOML examples for `lelabo train`.

## Official supervised golden paths

These are the supervised paths LeLabo treats as first-class:

1. `configs/train/supervised.quickstart.toml`
   `iris + mlp + bp`
2. Capsule path
   `lelabo create capsule my_capsule`, enable `optimizers/example.py`, then train `mnist + cnn + bp + capsule_sgd`
3. `configs/train/supervised.detailed.toml`
   `cifar10 + cnn + bp`
4. `configs/train/supervised.glue.toml`
   `glue/sst2 + bert + bp`
5. `configs/train/supervised.local_rule.toml`
   `mnist + mlp + dfa`

Each config should declare:

- `config_version`: use `"auto"` (or explicit schema version, currently `1.0`)
- `lelabo_version`: use `"auto"` (resolved from installed LeLabo version)

Priority order at runtime:

1. Library defaults (`src/lelabo/config/assets/*.toml`, loaded by `src/lelabo/config/defaults.py`)
2. Project TOML (`train.<mode>.toml`, `train.toml`, `configs/train.<mode>.toml`, `configs/train.toml`)
3. Explicit CLI overrides
4. Advanced `--set key=value` overrides

`--config` is optional. If provided, it overrides the auto-detected project config path.

Examples:

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml --set optimizer.params.lr=0.02
```

```bash
lelabo train supervised --config configs/train/supervised.glue.toml
```

```bash
lelabo train supervised --config configs/train/supervised.glue.toml --set hf.glue_task=stsb --loss mse --metrics mse,mae
```

```bash
lelabo train supervised --config configs/train/supervised.local_rule.toml
```

Built-in supervised metrics you can request in `[[metrics]]`:

- `accuracy` / `acc`
- `precision`, `recall`, `f1` (`average = macro|micro|weighted|binary`)
- `mse`, `mae`, `rmse`, `r2`

Example monitor:

```toml
[[callbacks]]
name = "earlystopping"
[callbacks.params]
monitor = "val.f1_macro"
```

For custom metric plugins inside a capsule, use `metrics/example.py` patterns:

1. Register a streaming metric with `register_metric`
2. Derive from `ClassificationStreamingMetric`, `RegressionStreamingMetric`, or `ScalarStreamingMetric`

For custom scheduler plugins inside a capsule, use `schedulers/example.py`
with `register_scheduler`.

For custom optimizer plugins inside a capsule, use `optimizers/example.py`
with `register_optimizer`.

For custom callbacks inside a capsule, use `callbacks/example.py`
with `register_callback`.

For the capsule optimizer golden path:

```bash
lelabo create capsule my_capsule
cd my_capsule
```

Enable the `capsule_sgd` example in `optimizers/example.py`, then run:

```bash
lelabo list optimizers
```

```bash
lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml
```

```bash
lelabo train rl --config configs/train/rl.detailed.toml --env CartPole-v1
```
