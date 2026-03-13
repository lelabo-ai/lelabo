"""
example.py

Purpose
-------
Add a custom trainer metric to this capsule.

Contract
--------
1. Register a builder with `@register_metric("example_error_rate", kind="classification")`
2. The builder receives a `MetricContext`
3. The builder returns a streaming `TrainerMetric`

Where params come from
----------------------
Metric params are namespaced by metric name and read with
`ctx.metric_params("example_error_rate")`.
For exact context fields, read `resources/LELABO_REFERENCE.md`.
For the param mapping, read `resources/PARAM_FLOW.md`.

Official example
----------------
`ExampleErrorRate` measures classification error from the confusion matrix.

Config snippet
--------------
[[metrics]]
name = "example_error_rate"

[metrics.params.example_error_rate]
as_percent = true

How to activate
---------------
Uncomment `@register_metric("example_error_rate", kind="classification")`.

How to test
-----------
lelabo list metrics
lelabo train supervised --config configs/train.supervised.detailed.toml --metrics example_error_rate

Common errors
-------------
- `compute()` / `finalize()` must return numeric scalars only.
- If the decorator stays commented, the metric will not be discoverable.
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


# @register_metric("example_error_rate", kind="classification")
def build_example_error_rate(ctx: MetricContext) -> ExampleErrorRate:
    params = ctx.metric_params("example_error_rate")
    return ExampleErrorRate(as_percent=bool(params.get("as_percent", False)))
