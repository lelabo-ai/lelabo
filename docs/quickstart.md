# Quickstart

## 1. Inspect the registries

```bash
lelabo list
```

```bash
lelabo list models
```

```bash
lelabo list update-rules
```

## 2. Run the first BP tabular baseline

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

This is the smallest official baseline:

- dataset: `iris`
- model: `mlp`
- update rule: `bp`

## 3. Run the official BP vision path

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml
```

This is the main vision baseline:

- dataset: `cifar10`
- model: `cnn`
- update rule: `bp`

## 4. Create a capsule and enable the optimizer path

```bash
lelabo capsule init my_capsule
cd my_capsule
```

Then:

1. open `optimizers/example.py`
2. uncomment `@register_optimizer("capsule_sgd")`
3. run `lelabo list optimizers`
4. run `lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml`

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

## 7. Override config values

```bash
lelabo train supervised \
  --config configs/train/supervised.detailed.toml \
  --set model.params.hidden=1024 \
  --set scheduler.params.gamma=0.5
```

Use `--set` for quick ablations without editing the TOML file.

## 8. Save a run directory

```bash
lelabo train supervised \
  --dataset iris \
  --model mlp \
  --rule bp \
  --run-dir outputs/runs/iris_bp_demo
```
