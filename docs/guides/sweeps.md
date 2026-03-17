# Run Parameter Sweeps

A sweep runs the same experiment across a grid of hyperparameters. LeLabo generates all combinations, executes them as independent `lelabo train` subprocesses, and writes structured artifacts for each run.

## Sweep config file

A sweep is defined by a YAML file with three sections:

```yaml
name: lr_seed_sweep
display_keys: [rule, dataset, lr, seed]

base:
  dataset: mnist
  model: mlp
  epochs: 30
  batch: 64
  device: auto
  optimizer: adamw
  lr: 1e-3

grid:
  rule: [bp, dfa]
  lr: [1e-2, 1e-3, 1e-4]
  seed: [0, 1, 2]
```

| Key | Required | Description |
|---|---|---|
| `name` | no | Experiment name. Used as output folder name and W&B group. Defaults to a timestamp. |
| `display_keys` | no | Parameters shown in run folder names. Makes `ls` readable. |
| `base` | no | Default arguments applied to every run. |
| `grid` | **yes** | Parameters to sweep. Each key maps to a list of values. LeLabo takes the cartesian product. |

The config above produces `2 rules × 3 lr × 3 seeds = 18` runs.

## Running a sweep

```bash
lelabo sweep run --config experiments/sweeps/lr_seed.yaml
```

Each job executes as:

```
python -m lelabo.cli.main train --dataset mnist --model mlp ... --run-dir outputs/runs/lr_seed_sweep/rule=bp__lr=0.01__seed=0__id=a3f1c2
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | required | Path to the sweep YAML config |
| `--outdir PATH` | `outputs/runs` | Parent directory for all sweep outputs |
| `--name NAME` | from config | Override the experiment name |
| `--max-parallel N` | `1` | Number of concurrent jobs |
| `--gpus 0,1,2` | — | GPU IDs for round-robin assignment |
| `--dry-run` | — | Print commands without executing |

### Dry run

Always preview before launching a large sweep:

```bash
lelabo sweep run --config sweep.yaml --dry-run
```

This prints every command that would be executed, without running anything.

### Parallel execution

Run 4 jobs at a time:

```bash
lelabo sweep run --config sweep.yaml --max-parallel 4
```

### Multi-GPU

Distribute jobs across GPUs with round-robin scheduling:

```bash
lelabo sweep run --config sweep.yaml --max-parallel 4 --gpus 0,1,2,3
```

Job 0 gets GPU 0, job 1 gets GPU 1, ..., job 4 gets GPU 0 again.

## Output structure

```
outputs/runs/lr_seed_sweep/
├── plan.json                          # Full sweep plan (all jobs + commands)
├── rule=bp__lr=0.01__seed=0__id=a3f1c2/
│   ├── stdout.log                     # Full stdout/stderr
│   ├── meta.json                      # Run metadata
│   ├── metrics.jsonl                  # Epoch-by-epoch metrics
│   ├── summary.json                   # Final results
│   └── resolved_config.yaml           # Exact config used
├── rule=bp__lr=0.01__seed=1__id=b7e2d4/
│   └── ...
└── ...
```

Each run directory follows the standard [run artifacts contract](../reference/run-artifacts.md). The `plan.json` at the root lists every job with its index, command, arguments, and directory.

### Run folder naming

Folder names are built from `display_keys` plus a deterministic short hash (`id=...`). The hash guarantees uniqueness even when display keys don't cover all grid parameters.

## W&B integration

When [W&B is configured](wandb.md), sweep runs are automatically grouped. LeLabo injects `--set wandb.group=<sweep_name>` into every job command, so all runs appear under the same group in the W&B dashboard.

```yaml
# sweep.yaml
name: compare_rules    # ← becomes the W&B group name

base:
  dataset: cifar10
  model: cnn
  epochs: 50

grid:
  rule: [bp, dfa, fa]
  seed: [0, 1, 2]
```

```bash
# Make sure W&B is configured
export WANDB_PROJECT=my-project

lelabo sweep run --config sweep.yaml --max-parallel 3
```

In the W&B UI, filter by group `compare_rules` to see all 9 runs side by side.

## Comparing results

### Quick shell comparison

```bash
for dir in outputs/runs/lr_seed_sweep/*/; do
  name=$(basename "$dir")
  acc=$(python -c "
import json
s = json.load(open('${dir}summary.json'))
print(f\"{s['best']['best_value']:.4f}\")
" 2>/dev/null || echo "N/A")
  echo "$name → $acc"
done
```

### Python analysis

```python
import json
from pathlib import Path

sweep_dir = Path("outputs/runs/lr_seed_sweep")
results = []

for run_dir in sorted(sweep_dir.iterdir()):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text())
    config = json.loads((run_dir / "meta.json").read_text())
    results.append({
        "run": run_dir.name,
        "best_val_acc": summary["best"]["best_value"],
        "epochs": summary["runtime"]["epochs_completed"],
        "args": config["args"],
    })

# Sort by best accuracy
results.sort(key=lambda r: r["best_val_acc"], reverse=True)
for r in results[:5]:
    print(f"{r['best_val_acc']:.4f}  {r['run']}")
```

## Sweep config in capsules

When you create a capsule with `lelabo capsule init`, a `sweeps/` directory is scaffolded with an example config:

```
my_capsule/
├── sweeps/
│   └── example.yaml
└── ...
```

This lets you version sweep configs alongside capsule code. Run them with:

```bash
lelabo sweep run --config sweeps/example.yaml
```

## Recipes

### Compare update rules across seeds

```yaml
name: rule_comparison
display_keys: [rule, seed]

base:
  dataset: mnist
  model: mlp
  epochs: 30
  batch: 64
  optimizer: sgd
  lr: 0.01

grid:
  rule: [bp, dfa, fa, drtp]
  seed: [0, 1, 2, 3, 4]
```

### Learning rate search

```yaml
name: lr_search
display_keys: [lr, seed]

base:
  dataset: cifar10
  model: cnn
  rule: bp
  epochs: 50
  batch: 128
  optimizer: adamw

grid:
  lr: [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
  seed: [0, 1, 2]
```

### Model × rule matrix

```yaml
name: model_rule_matrix
display_keys: [model, rule, dataset]

base:
  epochs: 30
  batch: 64
  optimizer: adamw
  lr: 1e-3
  seed: 42

grid:
  dataset: [mnist, cifar10]
  model: [mlp, cnn]
  rule: [bp, dfa, fa]
```

## Tips

- **Start with `--dry-run`** to verify the number of jobs and commands before launching.
- **Use `display_keys`** to keep folder names readable — include only the parameters that vary.
- **Set `--max-parallel`** to match your hardware. On a single GPU, keep it at 1. With multiple GPUs, match `--max-parallel` to `--gpus` count.
- **Check `plan.json`** if a sweep fails — it records every job's command for easy re-running.
- **Combine with W&B** for live monitoring. Each run logs independently, and the group view lets you compare curves in real time.
