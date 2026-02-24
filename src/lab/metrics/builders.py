from __future__ import annotations

from typing import Any

from .probes import BuiltinEpochMetric
from .registry import MetricContext, register_metric


_CLASSIF_PARAMS = {
    "average": "macro|micro|weighted|binary",
    "positive_label": "int (used when average='binary')",
}


def _metric_params(ctx: MetricContext, *names: str) -> dict[str, Any]:
    for name in names:
        params = ctx.metric_params(name)
        if params:
            return params
    return {}


def _metric_average(params: dict[str, Any], default: str = "macro") -> str:
    average = str(params.get("average", default)).strip().lower()
    if average not in {"macro", "micro", "weighted", "binary"}:
        return str(default)
    return average


def _positive_label(params: dict[str, Any]) -> int:
    try:
        return int(params.get("positive_label", 1))
    except Exception:
        return 1


def _output_key_for(metric: str, average: str) -> str:
    metric = str(metric).strip().lower()
    if metric == "accuracy":
        return "acc"
    if metric in {"precision", "recall", "f1"}:
        if average == "binary":
            return metric
        return f"{metric}_{average}"
    return metric


@register_metric(
    "accuracy",
    kind="classification",
    output_key="acc",
    description="Classification accuracy.",
    params={},
)
@register_metric(
    "acc",
    kind="classification",
    output_key="acc",
    description="Alias for accuracy.",
    params={},
)
def build_accuracy(ctx: MetricContext):
    params = _metric_params(ctx, "accuracy", "acc")
    return BuiltinEpochMetric(
        metric="accuracy",
        output_key="acc",
        average=_metric_average(params, default="macro"),
        positive_label=_positive_label(params),
    )


@register_metric(
    "precision",
    kind="classification",
    output_key="precision_<average>",
    description="Classification precision.",
    params=_CLASSIF_PARAMS,
)
def build_precision(ctx: MetricContext):
    params = _metric_params(ctx, "precision")
    average = _metric_average(params, default="macro")
    return BuiltinEpochMetric(
        metric="precision",
        output_key=_output_key_for("precision", average),
        average=average,
        positive_label=_positive_label(params),
    )


@register_metric(
    "precision_macro",
    kind="classification",
    output_key="precision_macro",
    description="Macro precision.",
    params={},
)
def build_precision_macro(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="precision", output_key="precision_macro", average="macro")


@register_metric(
    "precision_micro",
    kind="classification",
    output_key="precision_micro",
    description="Micro precision.",
    params={},
)
def build_precision_micro(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="precision", output_key="precision_micro", average="micro")


@register_metric(
    "precision_weighted",
    kind="classification",
    output_key="precision_weighted",
    description="Weighted precision.",
    params={},
)
def build_precision_weighted(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="precision", output_key="precision_weighted", average="weighted")


@register_metric(
    "recall",
    kind="classification",
    output_key="recall_<average>",
    description="Classification recall.",
    params=_CLASSIF_PARAMS,
)
def build_recall(ctx: MetricContext):
    params = _metric_params(ctx, "recall")
    average = _metric_average(params, default="macro")
    return BuiltinEpochMetric(
        metric="recall",
        output_key=_output_key_for("recall", average),
        average=average,
        positive_label=_positive_label(params),
    )


@register_metric(
    "recall_macro",
    kind="classification",
    output_key="recall_macro",
    description="Macro recall.",
    params={},
)
def build_recall_macro(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="recall", output_key="recall_macro", average="macro")


@register_metric(
    "recall_micro",
    kind="classification",
    output_key="recall_micro",
    description="Micro recall.",
    params={},
)
def build_recall_micro(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="recall", output_key="recall_micro", average="micro")


@register_metric(
    "recall_weighted",
    kind="classification",
    output_key="recall_weighted",
    description="Weighted recall.",
    params={},
)
def build_recall_weighted(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="recall", output_key="recall_weighted", average="weighted")


@register_metric(
    "f1",
    kind="classification",
    output_key="f1_<average>",
    description="F1 score.",
    params=_CLASSIF_PARAMS,
)
def build_f1(ctx: MetricContext):
    params = _metric_params(ctx, "f1")
    average = _metric_average(params, default="macro")
    return BuiltinEpochMetric(
        metric="f1",
        output_key=_output_key_for("f1", average),
        average=average,
        positive_label=_positive_label(params),
    )


@register_metric(
    "f1_macro",
    kind="classification",
    output_key="f1_macro",
    description="Macro F1 score.",
    params={},
)
def build_f1_macro(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="f1", output_key="f1_macro", average="macro")


@register_metric(
    "f1_micro",
    kind="classification",
    output_key="f1_micro",
    description="Micro F1 score.",
    params={},
)
def build_f1_micro(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="f1", output_key="f1_micro", average="micro")


@register_metric(
    "f1_weighted",
    kind="classification",
    output_key="f1_weighted",
    description="Weighted F1 score.",
    params={},
)
def build_f1_weighted(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="f1", output_key="f1_weighted", average="weighted")


@register_metric(
    "mse",
    kind="regression",
    output_key="mse",
    description="Mean squared error.",
    params={},
)
def build_mse(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="mse", output_key="mse")


@register_metric(
    "mae",
    kind="regression",
    output_key="mae",
    description="Mean absolute error.",
    params={},
)
def build_mae(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="mae", output_key="mae")


@register_metric(
    "rmse",
    kind="regression",
    output_key="rmse",
    description="Root mean squared error.",
    params={},
)
def build_rmse(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="rmse", output_key="rmse")


@register_metric(
    "r2",
    kind="regression",
    output_key="r2",
    description="R2 score.",
    params={},
)
def build_r2(ctx: MetricContext):
    _ = ctx
    return BuiltinEpochMetric(metric="r2", output_key="r2")
