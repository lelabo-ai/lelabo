# lab/core/callbacks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math
import torch


class Callback:
    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        pass

    def on_train_end(self, trainer: Any, logs: Dict[str, float]) -> None:
        pass


@dataclass
class EarlyStoppingConfig:
    monitor: str = "val.acc"          # e.g., "val.acc" or "val.loss"
    mode: str = "max"                 # "max" for acc, "min" for loss
    patience: int = 10
    min_delta: float = 0.0
    warmup_epochs: int = 0
    restore_best: bool = True


class EarlyStopping(Callback):
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best: float = -math.inf if cfg.mode == "max" else math.inf
        self.best_epoch: int = 0
        self.bad_epochs: int = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def _is_improvement(self, value: float) -> bool:
        if self.cfg.mode == "max":
            return value > (self.best + self.cfg.min_delta)
        return value < (self.best - self.cfg.min_delta)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        if epoch <= self.cfg.warmup_epochs:
            return

        if self.cfg.monitor not in logs:
            return

        value = float(logs[self.cfg.monitor])
        if self._is_improvement(value):
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            if self.cfg.restore_best:
                # store on CPU
                self.best_state = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.cfg.patience:
                trainer.stop_training = True
                trainer.stop_reason = (
                    f"EarlyStopping: no improvement in '{self.cfg.monitor}' "
                    f"for {self.cfg.patience} epochs (best={self.best:.6f} at epoch {self.best_epoch})."
                )

    def on_train_end(self, trainer: Any, logs: Dict[str, float]) -> None:
        if self.cfg.restore_best and self.best_state is not None:
            trainer.model.load_state_dict(self.best_state, strict=True)
