# Quickstart

## 1. Inspect available components

```bash
lelabo list
```

```bash
lelabo list models
```

```bash
lelabo list update-rules
```

## 2. Run a first supervised experiment

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

This is the simplest baseline:

- dataset: `iris`
- model: `mlp`
- update rule: `bp`

## 3. Run the official BP vision path

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml
```

This is the main vision baseline:

- dataset: `cifar10`
- model: built-in convnet (`cnn`)
- update rule: `bp`

## 4. Create a capsule and add a custom optimizer

```bash
lelabo create capsule my_capsule
cd my_capsule
```

Open `optimizers/example.py`, enable the `capsule_sgd` starter, then inspect discovery:

```bash
lelabo list optimizers
```

Run the capsule optimizer path:

```bash
lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml
```

This path proves that you can inject your own optimizer without forking the built-in package.

## 5. Run the official GLUE / BERT path

```bash
lelabo train supervised --config configs/train/supervised.glue.toml
```

This is the main NLP baseline:

- dataset: `glue/sst2`
- model: `bert`
- update rule: `bp`

## 6. Run the official local-rule path

```bash
lelabo train supervised --config configs/train/supervised.local_rule.toml
```

This is the main non-BP reference path:

- dataset: `mnist`
- model: `mlp`
- update rule: `dfa`

## 7. Override nested config values

The supervised CLI also supports auto-detection when no config is passed. It looks for:

- `train.supervised.toml`
- `train.toml`
- `configs/train.supervised.toml`
- `configs/train.toml`

```bash
lelabo train supervised \
  --config configs/train/supervised.detailed.toml \
  --set model.params.hidden=1024 \
  --set scheduler.params.gamma=0.5
```

Use `--set` when you want a quick ablation without editing the TOML file.
Structured values must be valid TOML, and strings with spaces or special characters should be quoted.

## 8. Save a run directory

```bash
lelabo train supervised \
  --dataset iris \
  --model mlp \
  --algo bp \
  --run-dir outputs/runs/iris_bp_demo
```

This writes run metadata and metrics files in the target directory.
