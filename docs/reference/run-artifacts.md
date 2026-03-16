# Run Artifacts Contract

Exact schemas for every file written to a run directory.

---

## `meta.json`

Schema version: `run_meta/v1`

Written at the start of a run with `status: "running"`. Updated to `"succeeded"`, `"failed"`, or `"interrupted"` when the run ends.

```json
{
  "schema_version": "run_meta/v1",
  "run_id": "string",
  "task": "supervised | rl",
  "status": "running | succeeded | failed | interrupted",
  "started_at": "ISO-8601 UTC timestamp",
  "finished_at": "ISO-8601 UTC timestamp | null",
  "error": "string | null",
  "args": {
    "dataset": "string",
    "model": "string",
    "rule": "string",
    "optimizer": "string",
    "epochs": "int",
    "batch": "int",
    "lr": "float",
    "device": "string",
    "seed": "int",
    "determinism": "string",
    "run_dir": "string"
  }
}
```

---

## `resolved_config.yaml`

The exact configuration used for this run, after merging library defaults, config file, CLI overrides, and `--set` overrides.

This file can be passed directly as `--config` to reproduce the run.

---

## `seeds.json`

Schema version: `run_seeds/v1`

```json
{
  "schema_version": "run_seeds/v1",
  "seed": "int",
  "mode": "off | relaxed | strict",
  "deterministic_algorithms": "bool"
}
```

---

## `metrics.jsonl`

One JSON object per line. Each line is an event emitted during training.

### Seed event

```json
{"t": "seed", "event": "seed", "seed": 2, "determinism": "relaxed", "deterministic_algorithms": false}
```

### Epoch event

```json
{
  "t": "epoch",
  "epoch": 1,
  "train.loss": 0.812,
  "train.acc": 0.61,
  "val.loss": 0.743,
  "val.acc": 0.68,
  "lr": 0.001
}
```

Metric keys follow the pattern `{split}.{metric_name}`. Additional metrics registered in `[[metrics]]` appear with the same pattern.

---

## `summary.json`

Schema version: `run_summary/v1`

Written when training completes.

```json
{
  "schema_version": "run_summary/v1",
  "run_id": "string",
  "status": "succeeded | failed | interrupted",
  "artifacts": {
    "meta": "meta.json",
    "config": "resolved_config.yaml",
    "seeds": "seeds.json",
    "metrics": "metrics.jsonl"
  },
  "best": {
    "source": "string",
    "monitor_name": "val.acc",
    "monitor_mode": "max | min",
    "best_value": "float",
    "epoch": "int",
    "train": { "split": "train", "loss": "float", "metric": "float | null", "scalars": {} },
    "val": { "split": "val", "loss": "float", "metric": "float | null", "scalars": {} }
  },
  "runtime": {
    "epochs_completed": "int",
    "total_train_time_sec": "float",
    "stopped_early": "bool",
    "stop_reason": "string | null"
  },
  "restoration": {
    "enabled": "bool",
    "best_epoch": "int | null",
    "best_checkpoint_available": "bool",
    "restored_on_train_end": "bool"
  }
}
```

---

## `checkpoints/`

Written only when `runtime.save_checkpoints = true`.

| File | Description |
|---|---|
| `last.pt` | Model `state_dict` at the end of the last completed epoch |
| `best.pt` | Model `state_dict` at the best monitored epoch (if available) |

Both files are PyTorch checkpoint files loadable with `torch.load()`.

Schema version: `run_checkpoint/v1`

```python
checkpoint = torch.load("checkpoints/last.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```
