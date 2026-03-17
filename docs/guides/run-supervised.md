# Run Supervised Experiments

## The five official paths

LeLabo ships with five reference configurations. They are the right starting points for any new experiment.

| Path | Command | Setup |
|---|---|---|
| BP tabular | `--config configs/train/supervised.quickstart.toml` | iris + mlp + bp |
| BP vision | `--config configs/train/supervised.detailed.toml` | cifar10 + cnn + bp |
| BP NLP | `--config configs/train/supervised.glue.toml` | glue/sst2 + bert + bp |
| Local rule | `--config configs/train/supervised.local_rule.toml` | mnist + mlp + dfa |
| Capsule optimizer | `--config configs/train.supervised.capsule_optimizer.toml` | mnist + cnn + bp + capsule_sgd |

Start with the tabular path — it has no GPU requirement and finishes in seconds.

## Swapping components

Any component can be swapped by name without editing the config file:

```bash
# Change the model
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --model cnn

# Change the update rule
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --rule dfa

# Change dataset, model, and rule at once
lelabo train supervised \
  --dataset mnist \
  --model mlp \
  --rule bp
```

To see what names are available:

```bash
lelabo list models
lelabo list update-rules
lelabo list datasets
lelabo list optimizers
```

## Overriding hyperparameters

Use CLI flags for the most common parameters:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --lr 0.0005 \
  --epochs 50 \
  --batch 64 \
  --weight-decay 1e-4
```

Use `--set` for any nested parameter:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --set model.params.hidden=512 \
  --set model.params.layers=4 \
  --set callbacks.0.params.patience=10
```

## Saving run artifacts

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --run-dir outputs/exp_001
```

LeLabo writes `meta.json`, `resolved_config.yaml`, `seeds.json`, `metrics.jsonl`, and `summary.json`. See [Run artifacts](../concepts/run-artifacts.md) for the full schema.

## Enabling checkpoints

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --run-dir outputs/exp_001 \
  --save-checkpoints
```

Writes `checkpoints/last.pt` and `checkpoints/best.pt` inside the run directory.

## Controlling determinism

```bash
# Relaxed (default) — sets seed, allows non-deterministic CUDA ops
lelabo train supervised --config ... --seed 42 --determinism relaxed

# Strict — fully deterministic, slower
lelabo train supervised --config ... --seed 42 --determinism strict

# Off — no seed, no determinism
lelabo train supervised --config ... --determinism off
```

## Console output

```bash
# Compact (default) — one line per epoch
lelabo train supervised --config ... --display compact

# Rich — progress bars and live metrics
lelabo train supervised --config ... --display rich

# Silent
lelabo train supervised --config ... --display none
```

## Reproducing a run

Every run directory contains a `resolved_config.yaml` that captures the exact configuration used. To reproduce:

```bash
lelabo train supervised \
  --config outputs/exp_001/resolved_config.yaml \
  --run-dir outputs/exp_001_repro
```

## Logging to Weights & Biases

Enable [W&B](https://wandb.ai) tracking by setting a project name:

```bash
export WANDB_PROJECT=my-research

lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --run-dir outputs/exp_001
```

Or configure it in the config file:

```toml
[wandb]
project = "my-research"
tags = ["baseline", "iris"]
```

Metrics are logged every epoch. The full resolved config is attached to the W&B run. See the [W&B integration guide](wandb.md) for all options.

## Running parameter sweeps

To run the same experiment across a grid of hyperparameters:

```yaml
# sweep.yaml
name: lr_search
base:
  dataset: iris
  model: mlp
  epochs: 20
  optimizer: adamw
grid:
  lr: [1e-2, 1e-3, 1e-4]
  seed: [0, 1, 2]
```

```bash
lelabo sweep run --config sweep.yaml
```

This generates 9 runs (3 lr × 3 seeds), each with its own run directory and artifacts. See the [sweep guide](sweeps.md) for parallel execution, multi-GPU, and W&B grouping.

## Forcing a device

```bash
lelabo train supervised --config ... --device cuda
lelabo train supervised --config ... --device cpu
lelabo train supervised --config ... --device mps
```

`--device auto` (default) picks CUDA if available, falls back to CPU.
