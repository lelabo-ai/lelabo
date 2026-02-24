from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.optim.lr_scheduler import ReduceLROnPlateau


class SchedulerController:
    """
    Thin wrapper around torch schedulers.

    It centralizes:
    - stepping interval (batch vs epoch)
    - monitor metric resolution for ReduceLROnPlateau
    """

    def __init__(
        self,
        scheduler: Any,
        *,
        interval: str = "epoch",
        monitor: str = "val.loss",
    ) -> None:
        self.scheduler = scheduler
        raw_interval = str(interval).strip().lower()
        self.interval = "batch" if raw_interval in {"batch", "step"} else "epoch"
        self.monitor = str(monitor).strip() or "val.loss"

    def step_batch(self, logs: Mapping[str, Any] | None = None) -> bool:
        if self.interval != "batch":
            return False
        return self._step(logs)

    def step_epoch(self, logs: Mapping[str, Any] | None = None) -> bool:
        if self.interval != "epoch":
            return False
        return self._step(logs)

    def _step(self, logs: Mapping[str, Any] | None = None) -> bool:
        if isinstance(self.scheduler, ReduceLROnPlateau):
            metric = self._pick_metric(logs)
            if metric is None:
                return False
            self.scheduler.step(metric)
            return True
        self.scheduler.step()
        return True

    def _pick_metric(self, logs: Mapping[str, Any] | None) -> float | None:
        if not logs:
            return None
        key = self.monitor
        if key in logs and isinstance(logs[key], (int, float)):
            return float(logs[key])
        for fallback in ("val.loss", "train.loss", "val.metric", "train.metric", "loss", "metric", "acc"):
            if fallback in logs and isinstance(logs[fallback], (int, float)):
                return float(logs[fallback])
        return None

    def state_dict(self) -> dict[str, Any]:
        if hasattr(self.scheduler, "state_dict"):
            return dict(self.scheduler.state_dict())
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(dict(state))

    def get_last_lr(self) -> list[float]:
        if hasattr(self.scheduler, "get_last_lr"):
            try:
                values = self.scheduler.get_last_lr()
                return [float(v) for v in values]
            except Exception:
                pass
        optimizer = getattr(self.scheduler, "optimizer", None)
        if optimizer is None:
            return []
        return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
