# Compare Update Rules

This tutorial shows how to run the same experiment with different update rules and compare results. This is one of the core values of LeLabo: same runtime, same model, same data, different learning — fair comparison by construction.

## Setup

We'll compare three rules on the same setup:

| Rule | What it does |
|---|---|
| `bp` | Standard backpropagation |
| `dfa` | Direct Feedback Alignment — random projections from output to each layer |
| `fa` | Feedback Alignment — random matrix replaces weight transpose in backprop |

All three will run on: **MNIST + MLP (256 hidden, 2 layers) + SGD (lr=0.01, momentum=0.9) + 20 epochs + seed 42**.

## Run all three

```bash
# Backpropagation
lelabo train supervised \
  --dataset mnist --model mlp --rule bp \
  --optimizer sgd --lr 0.01 \
  --epochs 20 --batch 64 --seed 42 \
  --run-dir outputs/compare/bp

# Direct Feedback Alignment
lelabo train supervised \
  --dataset mnist --model mlp --rule dfa \
  --optimizer sgd --lr 0.01 \
  --epochs 20 --batch 64 --seed 42 \
  --run-dir outputs/compare/dfa

# Feedback Alignment
lelabo train supervised \
  --dataset mnist --model mlp --rule fa \
  --optimizer sgd --lr 0.01 \
  --epochs 20 --batch 64 --seed 42 \
  --run-dir outputs/compare/fa
```

Same seed, same model, same optimizer, same data. The only variable is the update rule.

## Quick comparison

```bash
for rule in bp dfa fa; do
  echo "=== $rule ==="
  python -c "
import json
with open('outputs/compare/$rule/summary.json') as f:
    s = json.load(f)
best = s['best']
rt = s['runtime']
print(f\"  Best {best['monitor_name']}: {best['best_value']:.4f} (epoch {best['epoch']})\" )
print(f\"  Epochs: {rt['epochs_completed']}, stopped_early: {rt['stopped_early']}\")
print(f\"  Time: {rt['total_train_time_sec']:.1f}s\")
"
done
```

## Detailed comparison with Python

```python
import json
from pathlib import Path

rules = ["bp", "dfa", "fa"]
data = {}

for rule in rules:
    metrics_path = Path(f"outputs/compare/{rule}/metrics.jsonl")
    epochs = []
    with open(metrics_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("t") == "epoch":
                epochs.append(entry)
    data[rule] = epochs

# Print epoch-by-epoch comparison
header = f"{'Epoch':>5s}"
for rule in rules:
    header += f"  | {rule:>8s} val.acc"
print(header)
print("-" * len(header))

max_epochs = max(len(data[r]) for r in rules)
for i in range(max_epochs):
    line = f"{i+1:5d}"
    for rule in rules:
        if i < len(data[rule]):
            acc = data[rule][i].get("val.acc", 0)
            line += f"  | {acc:>16.4f}"
        else:
            line += f"  | {'—':>16s}"
    print(line)
```

## What to look for

**Convergence speed** — BP typically converges fastest. DFA and FA may need more epochs to reach comparable accuracy.

**Final accuracy** — BP usually wins on final accuracy, especially on harder tasks. DFA and FA close the gap on simpler datasets like MNIST.

**Training time** — local rules can be faster per step (no full backward pass), but the overhead of cache collection and manual gradient computation can offset this on small models.

**Stability** — some local rules are more sensitive to learning rate and initialization. If DFA or FA diverge, try reducing `--lr` or adjusting `feedback_scale`.

## Extending the comparison

Add more rules:

```bash
# DRTP
lelabo train supervised \
  --dataset mnist --model mlp --rule drtp \
  --optimizer sgd --lr 0.01 \
  --epochs 20 --batch 64 --seed 42 \
  --run-dir outputs/compare/drtp

# SoftHebb (uses deephebb model)
lelabo train supervised \
  --dataset mnist --model deephebb --rule softhebb \
  --optimizer sgd --lr 0.01 \
  --epochs 20 --batch 64 --seed 42 \
  --run-dir outputs/compare/softhebb
```

Try a harder dataset:

```bash
for rule in bp dfa fa; do
  lelabo train supervised \
    --dataset cifar10 --model cnn --rule $rule \
    --optimizer sgd --lr 0.01 \
    --epochs 50 --batch 64 --seed 42 \
    --run-dir outputs/compare_cifar/$rule
done
```

## What makes a fair comparison

A comparison is fair when the only variable is what you're testing. LeLabo makes this easy because:

- The same `Trainer` runs every rule — no hidden differences in training loops
- The same seed controls initialization and data ordering
- The same `metrics.jsonl` format makes results directly comparable
- The `resolved_config.yaml` documents exactly what ran

What can still break fairness:

- **Different optimal hyperparameters** — BP and DFA may have different optimal learning rates. A fairer comparison tunes each rule independently, then compares best results.
- **Model assumptions** — some rules require `declare_blocks()`, which constrains the model. If one rule uses a different model, the comparison is no longer apples-to-apples.
- **Random feedback initialization** — DFA and FA use random matrices that depend on the seed. Comparing across seeds is more robust than a single-seed comparison.
