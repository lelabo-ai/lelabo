from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainState:
    epoch: int = 0
    batch_idx: int = 0
    global_step: int = 0
    step: int = 0

    def set_step(self, value: int) -> None:
        v = int(value)
        self.global_step = v
        self.step = v

    def bump_step(self, n: int = 1) -> None:
        self.set_step(self.global_step + int(n))
