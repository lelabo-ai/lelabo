from __future__ import annotations

from typing import Any

from .registry import MetricContext


CLASSIFICATION_PARAMS = {
    "average": "macro|micro|weighted|binary",
    "positive_label": "int (used when average='binary')",
}


def metric_params(ctx: MetricContext, *names: str) -> dict[str, Any]:
    for name in names:
        params = ctx.metric_params(name)
        if params:
            return params
    return {}


def metric_average(params: dict[str, Any], default: str = "macro") -> str:
    average = str(params.get("average", default)).strip().lower()
    if average not in {"macro", "micro", "weighted", "binary"}:
        return str(default)
    return average


def positive_label(params: dict[str, Any]) -> int:
    raw = params.get("positive_label", 1)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid metric param 'positive_label'={raw!r}. Expected an integer."
        ) from exc


def output_key_for(metric: str, average: str) -> str:
    metric = str(metric).strip().lower()
    if metric == "accuracy":
        return "acc"
    if metric in {"precision", "recall", "f1"}:
        if average == "binary":
            return metric
        return f"{metric}_{average}"
    return metric
