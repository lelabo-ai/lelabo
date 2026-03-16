# Supervised Runtime

The supervised runtime is the core of LeLabo. It provides a clean training loop with explicit, predictable control flow — so you can focus on the method without rebuilding the infrastructure around it.

## The mental model

A supervised experiment has five moving parts:

```
Trainer
├── model          → the nn.Module being trained
├── learner        → owns train_step(), holds the update rule + optimizer
├── loss           → computes the training and eval loss
├── callbacks      → hook into the lifecycle (early stopping, logging…)
├── metrics        → stream per-batch scalars into per-epoch summaries
└── schedulers     → adjust the learning rate after each epoch or step
```

`Trainer.fit()` owns the epoch and batch loops. It calls the right pieces at the right moment. You don't subclass it — you configure it.

## Trainer

`Trainer` is the central runtime object. It:

- validates all dependencies at initialization
- runs the train loop (epoch → batch → `learner.train_step()`)
- runs evaluation passes on val or external loaders
- fires callback hooks at each lifecycle point
- collects metrics and scheduler updates
- writes structured logs through `RunLogger`
- returns a `FitResult` when training is complete

```python
from lelabo.core.trainer import Trainer

result = Trainer(
    model=model,
    learner=learner,
    loss=loss,
    callbacks=[early_stopping],
    metrics=[acc_metric],
    schedulers=[scheduler_ctrl],
    device="cuda",
).fit(train_loader, val_loader, epochs=30)
```

## Learner

The learner owns the update logic. It implements `train_step(batch, state)` and holds the optimizer and the update rule.

The trainer calls `learner.train_step()` on each batch. What happens inside — whether that's standard backprop, direct feedback alignment, or any other rule — is the learner's concern. The trainer only sees the returned scalars.

This separation is intentional: the trainer doesn't need to know how parameters are updated. It only needs a `train_step` it can call.

## Loss

The loss computes a scalar from model outputs and targets. It is called both during training (inside `train_step`) and during evaluation passes.

Built-in losses are registered by name and selected from the config:

```toml
[loss]
name = "ce"   # cross-entropy
```

## Callbacks

Callbacks hook into the training lifecycle without modifying the trainer itself. The base interface:

```python
class Callback:
    def on_train_start(self, trainer, state): ...
    def on_epoch_start(self, trainer, state): ...
    def on_batch_start(self, trainer, state): ...
    def on_batch_end(self, trainer, state, logs): ...
    def on_epoch_end(self, trainer, epoch_record, state): ...
    def on_eval_end(self, trainer, split_summary, state): ...
    def on_train_end(self, trainer, fit_result, state): ...
```

The built-in `EarlyStopping` callback monitors a scalar (e.g. `val.acc`) and can restore the best model state at the end of training.

```toml
[[callbacks]]
name = "earlystopping"
enabled = true
[callbacks.params]
monitor = "val.acc"
patience = 5
restore_best = true
```

## Metrics

Metrics are streaming objects that accumulate per-batch values and produce a scalar per epoch. They are separate from the loss — a run can compute accuracy, F1, or any custom scalar without that scalar being part of the optimization objective.

Metrics are selected from the config:

```toml
[[metrics]]
name = "acc"
```

## Schedulers

Schedulers adjust the learning rate according to a schedule. `SchedulerController` wraps a PyTorch scheduler and tells the trainer when to step it (after each epoch or each batch).

## TrainState

`TrainState` is the mutable shared state passed to all callbacks and hooks during a run. It tracks:

| Field | Description |
|---|---|
| `phase` | Current phase: `"train"`, `"eval"`, `"idle"` |
| `epoch` | Current epoch index |
| `global_step` | Total batches processed |
| `current_lr` | Current learning rate |
| `stop_requested` | Set by callbacks to trigger early stopping |
| `last_train` | `SplitSummary` from the last training epoch |
| `last_eval` | `SplitSummary` from the last evaluation pass |

## Output types

`Trainer.fit()` returns a `FitResult`. All output types are frozen dataclasses — they are immutable and JSON-serializable.

### `FitResult`

The top-level return value.

| Field | Type | Description |
|---|---|---|
| `history` | `list[EpochRecord]` | One record per completed epoch |
| `final_epoch` | `EpochRecord` | The last epoch |
| `best` | `BestSummary` | Best epoch according to the monitored metric |
| `runtime` | `FitRuntime` | Epochs completed, total time, early stop status |
| `restoration` | `RestorationStatus` | Whether best-state was restored at the end |

### `EpochRecord`

One completed epoch.

| Field | Type | Description |
|---|---|---|
| `epoch` | `int` | Epoch index |
| `train` | `SplitSummary` | Training split summary |
| `val` | `SplitSummary \| None` | Validation split summary |
| `lr` | `float \| None` | Learning rate at this epoch |
| `duration_sec` | `float` | Wall time for this epoch |

### `SplitSummary`

Aggregated metrics for one split over one epoch.

| Field | Type | Description |
|---|---|---|
| `split` | `str` | `"train"` or `"val"` |
| `loss` | `float` | Mean loss |
| `metric` | `float \| None` | Primary metric value (e.g. accuracy) |
| `scalars` | `dict[str, float]` | All additional metrics by name |
| `num_samples` | `int` | Samples processed |
| `duration_sec` | `float` | Wall time |

Scalars are accessible as `"train.loss"`, `"val.acc"`, etc. — which is the format used for monitoring, early stopping, and logging.

## What the runtime does not do

- It does not implement the learning algorithm — that is the learner's job.
- It does not modify model architecture — models are plain `nn.Module` objects.
- It does not manage multi-phase training programs as first-class objects yet. See [Research Notes](../research-notes.md) for the current limits.
