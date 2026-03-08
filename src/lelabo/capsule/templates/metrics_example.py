"""
example_metric.py

Custom metric template for capsule plugins.
"""

from __future__ import annotations

import torch

from lelabo.metrics import (
    ClassificationStreamingMetric,
    MetricContext,
    register_metric,
)


class ExampleErrorRate(ClassificationStreamingMetric):
    def __init__(self, *, output_key: str = "example_error_rate", as_percent: bool = False):
        super().__init__(output_key=output_key)
        self.as_percent = bool(as_percent)

    def compute_from_confusion_matrix(self, confusion_matrix: torch.Tensor) -> float | None:
        total = float(confusion_matrix.sum().item())
        if total <= 0.0:
            return None
        correct = float(confusion_matrix.diag().sum().item())
        error_rate = 1.0 - (correct / total)
        if self.as_percent:
            error_rate *= 100.0
        return float(error_rate)


@register_metric(
    "example_error_rate",
    kind="classification",
    params={"as_percent": "return the metric on a 0-100 scale instead of 0-1"},
)
def build_example_error_rate(ctx: MetricContext) -> ExampleErrorRate:
    params = ctx.metric_params("example_error_rate")
    return ExampleErrorRate(as_percent=bool(params.get("as_percent", False)))
