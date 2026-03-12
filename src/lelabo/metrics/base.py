from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TrainerMetric:
    """Optional training/evaluation metric aggregated by the Trainer."""

    name: str = "metric"

    def reset(
        self,
        split: str,
        state: Any | None = None,
    ) -> None:
        _ = (split, state)

    def update(
        self,
        split: str,
        stats: Mapping[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        _ = (split, stats, batch_size, state)

    def compute(
        self,
        split: str,
        state: Any | None = None,
    ) -> dict[str, float]:
        _ = (split, state)
        return {}

    def finalize(self, state: Any | None = None) -> dict[str, float]:
        _ = state
        return {}
