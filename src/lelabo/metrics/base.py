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
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        pass

    def on_epoch_end(self, trainer, epoch: int, state: Any | None = None) -> dict[str, float]:
        return {}

    def on_eval_start(
        self,
        trainer,
        split: str | None,
        state: Any | None = None,
    ) -> None:
        pass

    def on_eval_batch_end(
        self,
        trainer,
        split: str | None,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        pass

    def on_eval_end(
        self,
        trainer,
        split: str | None,
        state: Any | None = None,
    ) -> dict[str, float]:
        return {}

    def on_train_end(self, trainer, state: Any | None = None) -> dict[str, float]:
        return {}
