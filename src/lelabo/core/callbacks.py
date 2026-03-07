# lab/core/callbacks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math
import torch


class Callback:
    def on_train_start(self, trainer: Any, state: Any | None = None) -> None:
        pass

    def on_epoch_start(self, trainer: Any, state: Any | None = None) -> None:
        pass

    def on_batch_start(self, trainer: Any, state: Any | None = None) -> None:
        pass

    def on_batch_end(self, trainer: Any, state: Any | None = None, logs: Dict[str, float] | None = None) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], state: Any | None = None) -> None:
        pass

    def on_eval_end(self, trainer: Any, logs: Dict[str, float], state: Any | None = None) -> None:
        pass

    def on_train_end(self, trainer: Any, logs: Dict[str, float], state: Any | None = None) -> None:
        pass


@dataclass
class EarlyStoppingConfig:
    monitor: str = "val.acc"          # e.g., "val.acc" or "val.loss"
    mode: str = "max"                 # "max" for acc, "min" for loss
    patience: int = 10
    min_delta: float = 0.0
    min_delta_mode: str = "abs"       # "abs" | "rel"
    warmup_epochs: int = 0
    check_every_n_epochs: int = 1
    restore_best: bool = True
    restore_optimizer: bool = False
    restore_learner_state: bool = False
    restore_schedulers: bool = False
    restore_train_state: bool = False


class EarlyStopping(Callback):
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best: float = -math.inf if cfg.mode == "max" else math.inf
        self.best_epoch: int = 0
        self.bad_epochs: int = 0
        self.best_state: Optional[Dict[str, Any]] = None
        self._has_best: bool = False

    def _is_improvement(self, value: float) -> bool:
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

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float], state: Any | None = None) -> None:
        if self.cfg.check_every_n_epochs > 1 and (int(epoch) % int(self.cfg.check_every_n_epochs)) != 0:
            return
        if epoch <= self.cfg.warmup_epochs:
            return

        if self.cfg.monitor not in logs:
            return

        value = float(logs[self.cfg.monitor])
        if self._is_improvement(value):
            self.best = value
            self._has_best = True
            self.best_epoch = epoch
            self.bad_epochs = 0
            if self.cfg.restore_best:
                self.best_state = self._capture_state(trainer, self.cfg)
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.cfg.patience:
                trainer.stop_training = True
                trainer.stop_reason = (
                    f"EarlyStopping: no improvement in '{self.cfg.monitor}' "
                    f"for {self.cfg.patience} epochs (best={self.best:.6f} at epoch {self.best_epoch})."
                )

    def on_train_end(self, trainer: Any, logs: Dict[str, float], state: Any | None = None) -> None:
        if self.cfg.restore_best and self.best_state is not None:
            self._restore_state(trainer, self.best_state)
