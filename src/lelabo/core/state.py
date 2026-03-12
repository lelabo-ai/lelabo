from __future__ import annotations

from dataclasses import dataclass

from .train_types import EpochRecord, SplitSummary


@dataclass
class TrainState:
    phase: str = "idle"
    split: str | None = None
    epoch: int = 0
    batch_idx: int = 0
    global_step: int = 0
    step: int = 0
    current_lr: float | None = None
    train_samples_seen: int = 0
    eval_samples_seen: int = 0
    stop_requested: bool = False
    stop_reason: str | None = None
    last_train: SplitSummary | None = None
    last_eval: SplitSummary | None = None
    last_epoch: EpochRecord | None = None

    def set_step(self, value: int) -> None:
        v = int(value)
        self.global_step = v
        self.step = v

    def bump_step(self, n: int = 1) -> None:
        self.set_step(self.global_step + int(n))

    def request_stop(self, reason: str | None = None) -> None:
        self.stop_requested = True
        self.stop_reason = None if reason is None else str(reason)
