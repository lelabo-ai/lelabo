# Quickstart

This page gets you from a fresh install to a first working experiment in a few minutes.

## 1. Check what's available

Before running anything, inspect the built-in registries:

```bash
lelabo list models
lelabo list update-rules
lelabo list datasets
```

Every item in these lists can be selected by name in a config file or as a CLI flag.

## 2. Run your first experiment

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

This runs the smallest official baseline: **Iris + MLP + backpropagation**.

It is the right starting point because it has no GPU requirement, no large download, and finishes in seconds. If this runs, the full stack is working.

You should see a compact training log — one line per epoch with train loss, val loss, and accuracy.

## 3. Read the config

Open `configs/train/supervised.quickstart.toml` to see what just ran:

```toml
[dataset]
name = "iris"

[model]
name = "mlp"
[model.params]
hidden = 128
layers = 2

[update_rule]
name = "bp"

[optimizer]
name = "adamw"
[optimizer.params]
lr = 0.001

[train]
epochs = 20
batch = 32
```

Each block maps to a named component from the registry. Swap any `name` value to change the component — no code required.

## 4. Override from the CLI

You don't need to edit the config file for quick experiments. Use CLI flags:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --epochs 50 \
  --lr 0.01
```

For nested params, use `--set`:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --set model.params.hidden=256 \
  --set model.params.layers=3
```

## 5. Save a run directory

Add `--run-dir` to persist the outputs:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --run-dir outputs/my_first_run
```

LeLabo writes a structured set of files:

```
outputs/my_first_run/
├── meta.json            # run metadata, timestamps, status
├── resolved_config.yaml # exact config that was used
├── seeds.json           # random seed state
├── metrics.jsonl        # per-epoch metrics
└── summary.json         # final result summary
```

!!! tip "Reproducibility"
    The `resolved_config.yaml` captures every parameter after config + CLI merging. You can always recover exactly what was run.

## 6. Try the vision baseline

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml
```

This runs **CIFAR-10 + CNN + BP** — the main vision reference path. It will download the dataset on the first run.

## 7. Try a local learning rule

```bash
lelabo train supervised --config configs/train/supervised.local_rule.toml
```

This runs **MNIST + MLP + DFA** (Direct Feedback Alignment) — the first non-BP reference path.

## Next steps

- **Compare variants across seeds** → [Run parameter sweeps](../guides/sweeps.md)
- **Track and visualize results** → [Weights & Biases integration](../guides/wandb.md)
- **Extend LeLabo** with your own components → [Create a capsule](../guides/create-capsule.md)
- **Understand the runtime** → [Supervised runtime](../concepts/supervised-runtime.md)
- **See all CLI options** → [CLI reference](../reference/cli.md)
