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
lelabo train supervised --dataset iris --model mlp --algo bp
```

This is the simplest baseline:

- dataset: `iris`
- model: `mlp`
- update rule: `bp`

## 3. Run from a config file

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml
```

The supervised CLI also supports auto-detection when no config is passed. It looks for:

- `train.supervised.toml`
- `train.toml`
- `configs/train.supervised.toml`
- `configs/train.toml`

## 4. Override nested config values

```bash
lelabo train supervised \
  --config configs/train/supervised.detailed.toml \
  --set model.params.hidden=1024 \
  --set scheduler.params.gamma=0.5
```

Use `--set` when you want a quick ablation without editing the TOML file.
Structured values must be valid TOML, and strings with spaces or special characters should be quoted.

## 5. Save a run directory

```bash
lelabo train supervised \
  --dataset iris \
  --model mlp \
  --algo bp \
  --run-dir outputs/runs/iris_bp_demo
```

This writes run metadata and metrics files in the target directory.

## 6. Create a capsule

```bash
lelabo create capsule --name my_capsule
```

A capsule is the recommended way to add custom models, datasets, update rules, and metrics without modifying the built-in package directly.

## 7. Example supervised commands

CNN on CIFAR-10:

```bash
lelabo train supervised --dataset cifar10 --model cnn --algo bp
```

ResNet on CIFAR-10:

```bash
lelabo train supervised --dataset cifar10 --model resnet18 --algo bp
```

GLUE / BERT-style run:

```bash
lelabo train supervised --dataset glue --model bert --algo bp
```
