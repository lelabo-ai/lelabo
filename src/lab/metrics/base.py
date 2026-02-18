from __future__ import annotations

from typing import Any


class TrainingMetric:
    """Optional training-time metric hook."""

    name: str = "metric"

    def on_train_start(self, trainer, state: Any | None = None) -> None:
        pass

    def on_epoch_start(self, trainer, epoch: int, state: Any | None = None) -> None:
        pass

    def on_batch_end(
        self,
        trainer,
        stats: dict[str, float],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        pass

    def on_epoch_end(self, trainer, epoch: int, state: Any | None = None) -> dict[str, float]:
        return {}

    def on_train_end(self, trainer, state: Any | None = None) -> dict[str, float]:
        return {}
