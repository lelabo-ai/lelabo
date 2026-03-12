from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..metrics.payload import is_metric_payload_key
from .train_types import SplitSummary


class SplitAccumulator:
    def __init__(self, *, split: str | None):
        self.split = split
        self.loss_sum = 0.0
        self.scalar_sums: dict[str, float] = {}
        self.num_samples = 0
        self.num_batches = 0

    def update(self, *, loss: float, stats: Mapping[str, Any], batch_size: int) -> None:
        weight = int(max(1, int(batch_size)))
        self.loss_sum += float(loss) * float(weight)
        self.num_samples += weight
        self.num_batches += 1
        for key, value in stats.items():
            token = str(key)
            if token == "loss" or is_metric_payload_key(token):
                continue
            if isinstance(value, (int, float)):
                self.scalar_sums[token] = float(self.scalar_sums.get(token, 0.0) + (float(value) * float(weight)))

    def build(
        self,
        *,
        duration_sec: float | None,
        extra_scalars: Mapping[str, float] | None = None,
    ) -> SplitSummary:
        denom = float(max(1, self.num_samples))
        scalars = {str(k): float(v / denom) for k, v in self.scalar_sums.items()}
        if extra_scalars:
            for key, value in extra_scalars.items():
                if isinstance(value, (int, float)):
                    scalars[str(key)] = float(value)

        metric: float | None = None
        if isinstance(scalars.get("metric"), (int, float)):
            metric = float(scalars["metric"])
            scalars = {k: v for k, v in scalars.items() if k != "metric"}
        elif isinstance(scalars.get("acc"), (int, float)):
            metric = float(scalars["acc"])
        elif isinstance(scalars.get("agg"), (int, float)):
            metric = float(scalars["agg"])

        return SplitSummary(
            split=self.split,
            loss=float(self.loss_sum / denom),
            metric=metric,
            scalars=scalars,
            num_samples=int(self.num_samples),
            num_batches=int(self.num_batches),
            duration_sec=None if duration_sec is None else float(duration_sec),
        )

    def preview(
        self,
        *,
        extra_scalars: Mapping[str, float] | None = None,
    ) -> SplitSummary:
        return self.build(duration_sec=None, extra_scalars=extra_scalars)
