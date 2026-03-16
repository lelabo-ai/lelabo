"""Common payload keys passed from the trainer into metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


METRIC_Y_TRUE_KEY = "__metric_y_true__"
METRIC_Y_PRED_KEY = "__metric_y_pred__"
METRIC_KIND_KEY = "__metric_kind__"


@dataclass(frozen=True)
class MetricPayload:
    y_true: torch.Tensor
    y_pred: torch.Tensor
    kind: str


def is_metric_payload_key(key: str) -> bool:
    return str(key).startswith("__metric_")


def to_metric_tensor(value: Any) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    return value.detach().to("cpu")


def build_metric_payload(
    *,
    y_true: torch.Tensor | None,
    y_pred: torch.Tensor | None,
    kind: str,
) -> dict[str, Any]:
    true_t = to_metric_tensor(y_true)
    pred_t = to_metric_tensor(y_pred)
    if true_t is None or pred_t is None:
        return {}
    return {
        METRIC_Y_TRUE_KEY: true_t,
        METRIC_Y_PRED_KEY: pred_t,
        METRIC_KIND_KEY: str(kind),
    }


def extract_metric_payload(stats: Mapping[str, Any]) -> MetricPayload | None:
    y_true = to_metric_tensor(stats.get(METRIC_Y_TRUE_KEY))
    y_pred = to_metric_tensor(stats.get(METRIC_Y_PRED_KEY))
    if y_true is None or y_pred is None:
        return None
    kind = str(stats.get(METRIC_KIND_KEY, "")).strip().lower()
    if kind not in {"classification", "regression"}:
        return None
    return MetricPayload(y_true=y_true, y_pred=y_pred, kind=kind)
