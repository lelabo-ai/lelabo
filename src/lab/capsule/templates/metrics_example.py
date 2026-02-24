"""
example_metric.py

Custom metric template for capsule plugins.

This file intentionally shows a function metric with register_metric_fn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lab.metrics import (
    ClassificationMetricBase,
    MetricContext,
    register_metric,
    register_metric_fn,
)

# To register the metrics you need to uncomment the @register_metric_fn decorator and implement the function.
#@register_metric_fn(
#    "example_fn_accuracy",
#    kind="classification",
#)
def example_fn_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, metric_params: dict[str, Any]) -> float:
    _ = metric_params
    if y_true.numel() == 0 or y_pred.numel() == 0:
        return 0.0
    n = min(int(y_true.numel()), int(y_pred.numel()))
    return float((y_true[:n] == y_pred[:n]).float().mean().item())
