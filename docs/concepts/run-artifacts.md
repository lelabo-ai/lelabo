# Run Artifacts

When you pass `--run-dir` to `lelabo train`, LeLabo writes a structured set of files that capture everything about the run — the config that was used, the seed state, the per-epoch metrics, and the final result.

## Enable artifact writing

=== "CLI"

    ```bash
    lelabo train supervised \
      --config configs/train/supervised.quickstart.toml \
      --run-dir outputs/my_run
    ```

=== "Config"

    ```toml
    [runtime]
    run_dir = "outputs/my_run"
    ```

## Output structure

```
outputs/my_run/
├── meta.json
├── resolved_config.yaml
├── seeds.json
├── metrics.jsonl
├── summary.json
└── checkpoints/          ← only if save_checkpoints = true
    ├── last.pt
    └── best.pt
```

## Files

### `meta.json`

Run metadata written at the start and updated at the end.

```json
{
  "schema_version": "run_meta/v1",
  "run_id": "...",
  "task": "supervised",
  "status": "succeeded",
  "started_at": "2025-03-16T10:00:00+00:00",
  "finished_at": "2025-03-16T10:02:14+00:00",
  "args": { ... }
}
```

`status` is one of `"running"`, `"succeeded"`, `"failed"`, `"interrupted"`.

### `resolved_config.yaml`

The exact configuration that was used — after merging library defaults, the config file, CLI overrides, and `--set` overrides. This is the ground truth for what ran.

```yaml
dataset:
  name: iris
model:
  name: mlp
  params:
    hidden: 128
    layers: 2
update_rule:
  name: bp
optimizer:
  name: adamw
  params:
    lr: 0.001
runtime:
  seed: 2
  device: cpu
  ...
```

### `seeds.json`

The random seed state at the start of the run.

```json
{
  "schema_version": "run_seeds/v1",
  "seed": 2,
  "mode": "relaxed",
  "deterministic_algorithms": false
}
```

### `metrics.jsonl`

One JSON line per logged event. Includes per-epoch train/val metrics and any other events emitted during training.

```json
{"t": "epoch", "epoch": 1, "train.loss": 0.812, "val.loss": 0.743, "val.acc": 0.68}
{"t": "epoch", "epoch": 2, "train.loss": 0.631, "val.loss": 0.598, "val.acc": 0.74}
...
```

### `summary.json`

Final run summary written when training completes.

```json
{
  "schema_version": "run_summary/v1",
  "run_id": "...",
  "status": "succeeded",
  "best": {
    "epoch": 12,
    "monitor_name": "val.acc",
    "best_value": 0.953
  },
  "runtime": {
    "epochs_completed": 15,
    "stopped_early": true,
    "total_train_time_sec": 134.2
  }
}
```

## Checkpoints

Checkpoints are opt-in and require a `run_dir`:

=== "CLI"

    ```bash
    lelabo train supervised \
      --config configs/train/supervised.quickstart.toml \
      --run-dir outputs/my_run \
      --save-checkpoints
    ```

=== "Config"

    ```toml
    [runtime]
    run_dir = "outputs/my_run"
    save_checkpoints = true
    ```

When enabled:

- `checkpoints/last.pt` — model state at the end of the last epoch
- `checkpoints/best.pt` — model state at the best monitored epoch (if available)

!!! note
    Checkpoints save model state only. Optimizer and scheduler state are not included by default.

## Reproducibility

The `resolved_config.yaml` and `seeds.json` together capture everything needed to reproduce a run. To re-run with the exact same config:

```bash
lelabo train supervised --config outputs/my_run/resolved_config.yaml
```
