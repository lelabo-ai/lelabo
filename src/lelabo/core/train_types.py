from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SplitSummary:
    split: str | None
    loss: float
    metric: float | None
    scalars: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    num_batches: int = 0
    duration_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "loss": float(self.loss),
            "metric": None if self.metric is None else float(self.metric),
            "scalars": {str(k): float(v) for k, v in self.scalars.items()},
            "num_samples": int(self.num_samples),
            "num_batches": int(self.num_batches),
            "duration_sec": None if self.duration_sec is None else float(self.duration_sec),
        }

    def to_prefixed_scalars(self) -> dict[str, float]:
        prefix = str(self.split or "").strip()
        out: dict[str, float] = {}
        if prefix:
            out[f"{prefix}.loss"] = float(self.loss)
            if self.metric is not None:
                out[f"{prefix}.metric"] = float(self.metric)
            for key, value in self.scalars.items():
                out[f"{prefix}.{key}"] = float(value)
        else:
            out["loss"] = float(self.loss)
            if self.metric is not None:
                out["metric"] = float(self.metric)
            for key, value in self.scalars.items():
                out[str(key)] = float(value)
        return out


@dataclass(frozen=True)
class MonitorStatus:
    name: str
    mode: str
    best_value: float | None
    best_epoch: int | None
    improved_this_epoch: bool
    wait_epochs: int | None = None
    patience: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "mode": str(self.mode),
            "best_value": None if self.best_value is None else float(self.best_value),
            "best_epoch": None if self.best_epoch is None else int(self.best_epoch),
            "improved_this_epoch": bool(self.improved_this_epoch),
            "wait_epochs": None if self.wait_epochs is None else int(self.wait_epochs),
            "patience": None if self.patience is None else int(self.patience),
        }


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train: SplitSummary
    val: SplitSummary | None
    lr: float | None
    monitor: MonitorStatus | None
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "train": self.train.to_dict(),
            "val": None if self.val is None else self.val.to_dict(),
            "lr": None if self.lr is None else float(self.lr),
            "monitor": None if self.monitor is None else self.monitor.to_dict(),
            "duration_sec": float(self.duration_sec),
        }

    def to_log_values(self) -> dict[str, float]:
        out = self.train.to_prefixed_scalars()
        if self.val is not None:
            out.update(self.val.to_prefixed_scalars())
        if self.lr is not None:
            out["lr"] = float(self.lr)
        return out


@dataclass(frozen=True)
class RestorationStatus:
    restored_best_model: bool
    restored_best_epoch: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "restored_best_model": bool(self.restored_best_model),
            "restored_best_epoch": None if self.restored_best_epoch is None else int(self.restored_best_epoch),
        }


@dataclass(frozen=True)
class FitRuntime:
    epochs_completed: int
    total_train_time_sec: float
    stopped_early: bool
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_completed": int(self.epochs_completed),
            "total_train_time_sec": float(self.total_train_time_sec),
            "stopped_early": bool(self.stopped_early),
            "stop_reason": None if self.stop_reason is None else str(self.stop_reason),
        }


@dataclass(frozen=True)
class BestSummary:
    train_loss: float
    train_metric: float | None
    val_loss: float | None
    val_metric: float | None
    epoch_by_val: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": {
                "loss": float(self.train_loss),
                "metric": None if self.train_metric is None else float(self.train_metric),
            },
            "val": None if self.val_loss is None and self.val_metric is None else {
                "loss": None if self.val_loss is None else float(self.val_loss),
                "metric": None if self.val_metric is None else float(self.val_metric),
            },
            "epoch_by_val": None if self.epoch_by_val is None else int(self.epoch_by_val),
        }


@dataclass(frozen=True)
class FitResult:
    history: list[EpochRecord]
    final_epoch: EpochRecord
    best: BestSummary
    runtime: FitRuntime
    restoration: RestorationStatus
    run_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "history": [record.to_dict() for record in self.history],
            "final_epoch": self.final_epoch.to_dict(),
            "best": self.best.to_dict(),
            "runtime": self.runtime.to_dict(),
            "restoration": self.restoration.to_dict(),
        }
        if self.run_metrics:
            out["run_metrics"] = {str(k): float(v) for k, v in self.run_metrics.items()}
        return out
