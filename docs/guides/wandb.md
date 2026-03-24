# Weights & Biases Integration

LeLabo integrates with [Weights & Biases](https://wandb.ai) (W&B) for experiment tracking, metric visualization, and run comparison. The integration is optional — if `wandb` is not installed or not configured, training works exactly the same without it.

## Installation

W&B is an optional dependency:

```bash
pip install lelabo[wandb]
```

Or install it separately:

```bash
pip install wandb
```

Then authenticate:

```bash
wandb login
```

## Configuration

### Environment variables

The simplest way to enable W&B is via environment variables:

```bash
export WANDB_PROJECT=my-research
export WANDB_ENTITY=my-team      # optional, defaults to your personal account

lelabo train supervised --config configs/train/supervised.quickstart.toml --run-dir outputs/run_001
```

That's it. Every training run will now log to W&B under the project `my-research`.

### Config file

Add a `[wandb]` section to your TOML config:

```toml
[wandb]
project = "local-learning-rules"
entity = "my-team"
tags = ["dfa", "mnist", "baseline"]
group = "experiment-v2"
notes = "Testing DFA convergence with different learning rates"
enabled = true
```

### CLI overrides

Override any W&B setting with `--set`:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --set wandb.project=my-project \
  --set wandb.tags=dfa,cifar10 \
  --set wandb.group=lr-search \
  --set wandb.notes="Quick test" \
  --run-dir outputs/run_001
```

### Disabling W&B

If W&B is configured but you want to skip it for a specific run:

```bash
lelabo train supervised --config ... --set wandb.enabled=false
```

Or unset the environment variable:

```bash
unset WANDB_PROJECT
```

## Resolution order

W&B settings follow the same resolution order as all LeLabo config:

```
1. Library defaults (disabled)
2. Environment variables (WANDB_PROJECT, WANDB_ENTITY)
3. Config file [wandb] section
4. CLI --set overrides
```

Config file values take precedence over environment variables. CLI overrides take precedence over everything.

!!! warning "Config file beats environment variables"
    This is the opposite of the usual convention. If `WANDB_PROJECT` is exported in your shell but `project = "..."` is also set in the TOML config file, the **config file value wins**. To let the environment variable take effect, either remove the `project` key from the config, or use `--set wandb.project=...` at the CLI.

## What gets logged

### Metrics

Every numeric value in the epoch record is sent to W&B at each epoch:

- `train.loss`, `train.acc`, `train.f1`, ...
- `val.loss`, `val.acc`, `val.f1`, ...
- Any custom metric registered via `@register_metric`

### Config

The public run arguments are attached to the W&B run config: dataset, model, rule, optimizer, lr, epochs, seed, and other top-level parameters. This lets you filter, sort, and group runs by any of these values in the dashboard.

!!! note
    The **full** `resolved_config.yaml` (including nested model and optimizer params) is stored locally in `run_dir` but is not uploaded to W&B. Use `--set` overrides or `display_keys` in sweeps to surface additional params in the W&B UI.

### Summary

At the end of training, the final summary (best metrics, epoch count, training time) is pushed to the W&B run summary for easy table-based comparison.

## W&B config reference

| Key | Type | Default | Description |
|---|---|---|---|
| `wandb.project` | str | `None` | W&B project name. **Required to enable logging.** Falls back to `WANDB_PROJECT`. |
| `wandb.entity` | str | `None` | W&B team or user. Falls back to `WANDB_ENTITY`. |
| `wandb.tags` | list | `[]` | Tags for filtering runs in the dashboard. |
| `wandb.group` | str | `None` | Group name for organizing related runs (e.g., a sweep). |
| `wandb.notes` | str | `""` | Free-text notes attached to the run. |
| `wandb.enabled` | bool | `true` | Set to `false` to disable W&B even when project is set. |

## Using W&B with sweeps

If `WANDB_PROJECT` is configured, **every job in a `lelabo sweep` logs to W&B automatically** — no additional configuration required. LeLabo injects `--set wandb.group=<sweep_name>` into every subprocess, so all runs are grouped together in the dashboard.

```yaml
# sweep.yaml
name: compare_rules    # ← becomes the W&B group name

base:
  dataset: mnist
  model: mlp
  epochs: 30

grid:
  rule: [bp, dfa, fa]
  seed: [0, 1, 2]
```

```bash
export WANDB_PROJECT=my-project
lelabo sweep run --config sweep.yaml
```

All 9 runs appear under group `compare_rules` in the W&B dashboard. Use the group view to overlay training curves and compare final metrics.

!!! note "You don't need `wandb sweep`"
    LeLabo handles all sweep orchestration locally. If W&B is configured, runs appear in W&B automatically. There is no need to use `wandb sweep` or `wandb agent` — those are W&B's own distributed sweep system and are not used by LeLabo.

See [Run parameter sweeps](sweeps.md) for the full sweep guide.

## Dashboard tips

### Useful views

- **Group by `wandb.group`** to see sweep runs together
- **Sort by `summary.best.best_value`** to find the best hyperparameters
- **Filter by tags** to isolate specific experiments
- **Use parallel coordinates** to visualize hyperparameter interactions

### Comparing runs

1. Select runs in the table
2. Click "Compare" to overlay training curves
3. Use the "Diff" tab to see which config values differ

### Example queries

```
# W&B filter syntax
group == "lr_search" AND config.rule == "dfa"
tags IN ["baseline", "mnist"]
summary.best.best_value > 0.95
```

## Offline mode

If you're training on a machine without internet access, W&B supports offline mode:

```bash
export WANDB_MODE=offline

lelabo train supervised --config ... --run-dir outputs/run_001
```

Logs are saved locally by the wandb library. By default, wandb creates a `wandb/` directory in your **current working directory** at the time of the run. Sync later with:

```bash
wandb sync wandb/
```

## Graceful degradation

LeLabo handles W&B failures gracefully:

- **`wandb` not installed** — training proceeds normally, no error
- **`wandb.project` not set** — W&B is silently skipped
- **Network error during logging** — metric is dropped, training continues
- **`wandb.init()` fails** — logged as a warning, training continues

You never need to worry about W&B breaking a training run.
