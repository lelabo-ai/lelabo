from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .base import TrainingMetric


def _safe_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


class BPAlignmentEpochMetric(TrainingMetric):
    """
    Epoch-level BP alignment metric.

    Priority order:
    1) weighted mean of per-batch stats (if present in `train_step` output)
    2) learner.alignment_summary() epoch payload
    """

    def __init__(
        self,
        *,
        name: str,
        align_prefix: str,
        batch_stat_key: str,
        invert: bool = False,
    ):
        self.name = str(name)
        self.align_prefix = str(align_prefix)
        self.batch_stat_key = str(batch_stat_key)
        self.invert = bool(invert)
        self._sum = 0.0
        self._count = 0.0
        self._last_value: float | None = None

    def on_epoch_start(self, trainer, epoch: int, state: Any | None = None) -> None:
        self._sum = 0.0
        self._count = 0.0

    def on_batch_end(
        self,
        trainer,
        stats: dict[str, float],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        value = _safe_float(stats.get(self.batch_stat_key))
        if value is None:
            return
        w = float(max(1, int(batch_size)))
        self._sum += value * w
        self._count += w

    def on_epoch_end(self, trainer, epoch: int, state: Any | None = None) -> dict[str, float]:
        value = self._value_from_batches()
        if value is None:
            value = self._value_from_learner(trainer=trainer, epoch=epoch)
        if value is None:
            return {}

        if self.invert:
            value = 1.0 - value
        self._last_value = float(value)
        return {self.name: self._last_value}

    def on_train_end(self, trainer, state: Any | None = None) -> dict[str, float]:
        if self._last_value is None:
            return {}
        return {self.name: float(self._last_value)}

    def _value_from_batches(self) -> float | None:
        if self._count <= 0.0:
            return None
        return float(self._sum / self._count)

    def _value_from_learner(self, *, trainer, epoch: int) -> float | None:
        learner = getattr(trainer, "learner", None)
        if learner is None:
            return None

        summary_fn = getattr(learner, "alignment_summary", None)
        if not callable(summary_fn):
            return None

        payload = summary_fn()
        if not isinstance(payload, Mapping):
            return None

        epoch_means_key = f"{self.align_prefix}_epoch_means"
        epoch_means = payload.get(epoch_means_key)
        if isinstance(epoch_means, Mapping):
            direct = _safe_float(epoch_means.get(str(int(epoch))))
            if direct is not None:
                return direct
        return None
