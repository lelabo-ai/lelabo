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
  "run_id": "run_20260317_143052_a7b3e1",
  "task": "supervised",
  "status": "succeeded",
  "started_at": "2026-03-17T14:30:52+00:00",
  "finished_at": "2026-03-17T14:31:08+00:00",
  "error": null,
  "args": {
    "dataset": "iris",
    "model": "mlp",
    "rule": "bp",
    "optimizer": "adamw",
    "epochs": 20,
    "batch": 32,
    "lr": 0.001,
    "device": "cpu",
    "seed": 2,
    "determinism": "relaxed",
    "run_dir": "outputs/iris_bp"
  }
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
{"t": "seed", "event": "seed", "seed": 2, "determinism": "relaxed", "deterministic_algorithms": false}
{"t": "epoch", "epoch": 1, "train.loss": 1.098, "train.acc": 0.333, "val.loss": 1.054, "val.acc": 0.400, "lr": 0.001}
{"t": "epoch", "epoch": 2, "train.loss": 0.912, "train.acc": 0.533, "val.loss": 0.874, "val.acc": 0.600, "lr": 0.001}
{"t": "epoch", "epoch": 3, "train.loss": 0.743, "train.acc": 0.667, "val.loss": 0.698, "val.acc": 0.733, "lr": 0.001}
{"t": "epoch", "epoch": 4, "train.loss": 0.601, "train.acc": 0.790, "val.loss": 0.562, "val.acc": 0.800, "lr": 0.001}
{"t": "epoch", "epoch": 5, "train.loss": 0.487, "train.acc": 0.857, "val.loss": 0.451, "val.acc": 0.867, "lr": 0.001}
{"t": "epoch", "epoch": 6, "train.loss": 0.398, "train.acc": 0.905, "val.loss": 0.369, "val.acc": 0.933, "lr": 0.001}
{"t": "epoch", "epoch": 7, "train.loss": 0.331, "train.acc": 0.933, "val.loss": 0.308, "val.acc": 0.933, "lr": 0.001}
{"t": "epoch", "epoch": 8, "train.loss": 0.281, "train.acc": 0.943, "val.loss": 0.263, "val.acc": 0.933, "lr": 0.001}
{"t": "epoch", "epoch": 9, "train.loss": 0.243, "train.acc": 0.952, "val.loss": 0.229, "val.acc": 0.933, "lr": 0.001}
{"t": "epoch", "epoch": 10, "train.loss": 0.213, "train.acc": 0.952, "val.loss": 0.203, "val.acc": 0.953, "lr": 0.001}
```

### `summary.json`

Final run summary written when training completes.

```json
{
  "schema_version": "run_summary/v1",
  "run_id": "run_20260317_143052_a7b3e1",
  "status": "succeeded",
  "artifacts": {
    "meta": "meta.json",
    "config": "resolved_config.yaml",
    "seeds": "seeds.json",
    "metrics": "metrics.jsonl"
  },
  "best": {
    "source": "earlystopping",
    "monitor_name": "val.acc",
    "monitor_mode": "max",
    "best_value": 0.953,
    "epoch": 10,
    "train": { "split": "train", "loss": 0.213, "metric": 0.952, "scalars": {} },
    "val": { "split": "val", "loss": 0.203, "metric": 0.953, "scalars": {} }
  },
  "runtime": {
    "epochs_completed": 15,
    "total_train_time_sec": 16.4,
    "stopped_early": true,
    "stop_reason": "earlystopping: val.acc did not improve for 5 epochs"
  },
  "restoration": {
    "enabled": true,
    "best_epoch": 10,
    "best_checkpoint_available": false,
    "restored_on_train_end": true
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
