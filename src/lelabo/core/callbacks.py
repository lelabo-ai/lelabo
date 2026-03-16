"""Callback contracts and built-in callback implementations for the trainer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional
import math
import torch

from .train_types import EpochRecord, FitResult, MonitorStatus, RestorationStatus, SplitSummary


class Callback:
    """Base callback interface for trainer lifecycle hooks."""

    def on_train_start(self, trainer: Any, state: Any | None = None) -> None:
        pass

    def on_epoch_start(self, trainer: Any, state: Any | None = None) -> None:
        pass

    def on_batch_start(self, trainer: Any, state: Any | None = None) -> None:
        pass

    def on_batch_end(self, trainer: Any, state: Any | None = None, logs: Dict[str, Any] | None = None) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch_record: EpochRecord, state: Any | None = None) -> None:
        pass

    def on_eval_end(self, trainer: Any, split_summary: SplitSummary, state: Any | None = None) -> None:
        pass

    def on_train_end(self, trainer: Any, fit_result: FitResult, state: Any | None = None) -> None:
        pass


@dataclass
class EarlyStoppingConfig:
    """Configuration for the built-in early-stopping callback."""

    monitor: str = "val.acc"          # e.g., "val.acc" or "val.loss"
    mode: str = "auto"                # "auto" | "max" | "min"
    patience: int = 5
    min_delta: float = 0.0
    min_delta_mode: str = "abs"       # "abs" | "rel"
    warmup_epochs: int = 5
    check_every_n_epochs: int = 1
    restore_best: bool = True
    restore_optimizer: bool = False
    restore_learner_state: bool = False
    restore_schedulers: bool = False
    restore_train_state: bool = False


class EarlyStopping(Callback):
    """Stop training when a monitored scalar stops improving."""

    def __init__(self, cfg: EarlyStoppingConfig):
        normalized_mode = self._normalize_mode(cfg.mode, cfg.monitor)
        self.cfg = replace(cfg, mode=normalized_mode)
        self.best: float = -math.inf if self.cfg.mode == "max" else math.inf
        self.best_epoch: int = 0
        self.bad_epochs: int = 0
        self.best_state: Optional[Dict[str, Any]] = None
        self._has_best: bool = False
        self._improved_this_epoch: bool = False
        self._restored_on_train_end: bool = False

    def on_train_start(self, trainer: Any, state: Any | None = None) -> None:
        """Reset internal best-state tracking at the beginning of a fit call."""
        _ = (trainer, state)
        self.best = -math.inf if self.cfg.mode == "max" else math.inf
        self.best_epoch = 0
        self.bad_epochs = 0
        self.best_state = None
        self._has_best = False
        self._improved_this_epoch = False
        self._restored_on_train_end = False

    @staticmethod
    def _normalize_mode(raw_mode: Any, monitor: str) -> str:
        """Resolve ``auto`` monitor mode into ``min`` or ``max``."""
        mode = str(raw_mode if raw_mode is not None else "auto").strip().lower()
        if mode == "auto":
            key = str(monitor).strip().lower()
            if any(token in key for token in ("acc", "f1", "precision", "recall", "r2", "auc")):
                return "max"
            return "min"
        if mode not in {"max", "min"}:
            raise ValueError(f"Unsupported earlystopping mode '{raw_mode}'. Expected: auto|min|max.")
        return mode

    def _is_improvement(self, value: float) -> bool:
        """Return whether ``value`` improves on the current best according to config."""
        if not self._has_best:
            return True
        delta = float(self.cfg.min_delta)
        if str(self.cfg.min_delta_mode).strip().lower() == "rel":
            delta = abs(float(self.best)) * delta
        if self.cfg.mode == "max":
            return value > (self.best + delta)
        return value < (self.best - delta)

    @staticmethod
    def _to_cpu_state(raw: Any) -> Any:
        """Detach tensors recursively so a checkpoint snapshot is device-agnostic."""
        if torch.is_tensor(raw):
            return raw.detach().cpu().clone()
        if isinstance(raw, dict):
            return {str(k): EarlyStopping._to_cpu_state(v) for k, v in raw.items()}
        if isinstance(raw, list):
            return [EarlyStopping._to_cpu_state(v) for v in raw]
        if isinstance(raw, tuple):
            return tuple(EarlyStopping._to_cpu_state(v) for v in raw)
        return raw

    @staticmethod
    def _capture_state(trainer: Any, cfg: EarlyStoppingConfig) -> Dict[str, Any]:
        """Capture the subset of trainer state needed for best-state restoration."""
        payload: Dict[str, Any] = {
            "model": {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()},
        }

        if cfg.restore_optimizer:
            learner = getattr(trainer, "learner", None)
            optimizer = getattr(learner, "optimizer", None)
            if isinstance(optimizer, torch.optim.Optimizer):
                payload["optimizer"] = optimizer.state_dict()
        if cfg.restore_learner_state:
            learner = getattr(trainer, "learner", None)
            state_dict = getattr(learner, "state_dict", None)
            if callable(state_dict):
                payload["learner"] = EarlyStopping._to_cpu_state(state_dict())

        if cfg.restore_schedulers:
            sched_states: list[dict[str, Any] | None] = []
            for sched in list(getattr(trainer, "schedulers", []) or []):
                state_dict = getattr(sched, "state_dict", None)
                if callable(state_dict):
                    sched_states.append(dict(state_dict()))
                else:
                    sched_states.append(None)
            payload["schedulers"] = sched_states

        if cfg.restore_train_state:
            state = getattr(trainer, "state", None)
            if state is not None:
                payload["train_state"] = {
                    "epoch": int(getattr(state, "epoch", 0)),
                    "batch_idx": int(getattr(state, "batch_idx", 0)),
                    "global_step": int(getattr(state, "global_step", 0)),
                    "step": int(getattr(state, "step", 0)),
                }
        return payload

    @staticmethod
    def _restore_state(trainer: Any, payload: Dict[str, Any]) -> None:
        """Restore a previously captured best-state payload into the trainer."""
        model_state = payload.get("model")
        if isinstance(model_state, dict):
            trainer.model.load_state_dict(model_state, strict=True)

        optimizer_state = payload.get("optimizer")
        if isinstance(optimizer_state, dict):
            learner = getattr(trainer, "learner", None)
            optimizer = getattr(learner, "optimizer", None)
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.load_state_dict(optimizer_state)

        learner_state = payload.get("learner")
        learner = getattr(trainer, "learner", None)
        load_learner_state = getattr(learner, "load_state_dict", None)
        if isinstance(learner_state, dict) and callable(load_learner_state):
            load_learner_state(learner_state)

        sched_states = payload.get("schedulers")
        if isinstance(sched_states, list):
            schedulers = list(getattr(trainer, "schedulers", []) or [])
            for idx, state in enumerate(sched_states):
                if idx >= len(schedulers) or not isinstance(state, dict):
                    continue
                load_state = getattr(schedulers[idx], "load_state_dict", None)
                if callable(load_state):
                    load_state(state)

        train_state_payload = payload.get("train_state")
        state = getattr(trainer, "state", None)
        if isinstance(train_state_payload, dict) and state is not None:
            for key in ("epoch", "batch_idx", "global_step", "step"):
                if key in train_state_payload:
                    setattr(state, key, int(train_state_payload[key]))

    def _logs_from_epoch_record(self, epoch_record: EpochRecord) -> dict[str, float]:
        """Expose epoch scalars through the same naming scheme used by logging/monitoring."""
        return epoch_record.to_log_values()

    def status(self) -> MonitorStatus:
        """Return a snapshot of the current monitor/best-value state."""
        return MonitorStatus(
            name=str(self.cfg.monitor),
            mode=str(self.cfg.mode),
            best_value=(None if not self._has_best else float(self.best)),
            best_epoch=(None if not self._has_best else int(self.best_epoch)),
            improved_this_epoch=bool(self._improved_this_epoch),
            wait_epochs=int(self.bad_epochs),
            patience=int(self.cfg.patience),
        )

    def restoration_status(self) -> RestorationStatus:
        """Return whether best-state restoration is enabled and/or has happened."""
        best_epoch = int(self.best_epoch) if self._has_best and self.best_epoch > 0 else None
        return RestorationStatus(
            enabled=bool(self.cfg.restore_best),
            best_epoch=best_epoch,
            best_checkpoint_available=bool(self.best_state is not None),
            restored_on_train_end=bool(self._restored_on_train_end),
        )

    def on_epoch_end(self, trainer: Any, epoch_record: EpochRecord, state: Any | None = None) -> None:
        """Inspect the monitored scalar and request training stop when patience is exceeded."""
        epoch = int(epoch_record.epoch)
        self._improved_this_epoch = False
        if self.cfg.check_every_n_epochs > 1 and (epoch % int(self.cfg.check_every_n_epochs)) != 0:
            return
        if epoch <= self.cfg.warmup_epochs:
            return

        logs = self._logs_from_epoch_record(epoch_record)
        if self.cfg.monitor not in logs:
            return

        value = float(logs[self.cfg.monitor])
        if self._is_improvement(value):
            self.best = value
            self._has_best = True
            self.best_epoch = epoch
            self.bad_epochs = 0
            self._improved_this_epoch = True
            if self.cfg.restore_best:
                self.best_state = self._capture_state(trainer, self.cfg)
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.cfg.patience:
                reason = (
                    f"EarlyStopping: no improvement in '{self.cfg.monitor}' "
                    f"for {self.cfg.patience} epochs (best={self.best:.6f} at epoch {self.best_epoch})."
                )
                request_stop = getattr(trainer, "request_stop", None)
                if callable(request_stop):
                    request_stop(reason)
                else:
                    trainer.stop_training = True
                    trainer.stop_reason = reason
                if state is not None and hasattr(state, "request_stop"):
                    state.request_stop(reason)

    def on_train_end(self, trainer: Any, fit_result: FitResult, state: Any | None = None) -> None:
        """Restore the best captured state at the end of training when configured."""
        _ = (fit_result, state)
        if self.cfg.restore_best and self.best_state is not None:
            self._restore_state(trainer, self.best_state)
            self._restored_on_train_end = True
