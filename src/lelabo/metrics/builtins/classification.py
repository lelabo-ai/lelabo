from __future__ import annotations

from ..common import (
    CLASSIFICATION_PARAMS,
    metric_average,
    metric_params,
    output_key_for,
    positive_label,
)
from ..helpers import BuiltinStreamingMetric
from ..registry import MetricContext, register_metric


@register_metric("acc", kind="classification")
def build_accuracy(ctx: MetricContext):
    params = metric_params(ctx, "accuracy", "acc")
    return BuiltinStreamingMetric(
        metric="accuracy",
        output_key="acc",
        average=metric_average(params, default="macro"),
        positive_label=positive_label(params),
    )


@register_metric("precision", kind="classification", params=CLASSIFICATION_PARAMS)
def build_precision(ctx: MetricContext):
    params = metric_params(ctx, "precision")
    average = metric_average(params, default="macro")
    return BuiltinStreamingMetric(
        metric="precision",
        output_key=output_key_for("precision", average),
        average=average,
        positive_label=positive_label(params),
    )


@register_metric("precision_macro", kind="classification")
def build_precision_macro(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="precision", output_key="precision_macro", average="macro")


@register_metric("precision_micro", kind="classification")
def build_precision_micro(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="precision", output_key="precision_micro", average="micro")


@register_metric("precision_weighted", kind="classification")
def build_precision_weighted(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="precision", output_key="precision_weighted", average="weighted")


@register_metric("recall", kind="classification", params=CLASSIFICATION_PARAMS)
def build_recall(ctx: MetricContext):
    params = metric_params(ctx, "recall")
    average = metric_average(params, default="macro")
    return BuiltinStreamingMetric(
        metric="recall",
        output_key=output_key_for("recall", average),
        average=average,
        positive_label=positive_label(params),
    )


@register_metric("recall_macro", kind="classification")
def build_recall_macro(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="recall", output_key="recall_macro", average="macro")


@register_metric("recall_micro", kind="classification")
def build_recall_micro(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="recall", output_key="recall_micro", average="micro")


@register_metric("recall_weighted", kind="classification")
def build_recall_weighted(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="recall", output_key="recall_weighted", average="weighted")


@register_metric("f1", kind="classification", params=CLASSIFICATION_PARAMS)
def build_f1(ctx: MetricContext):
    params = metric_params(ctx, "f1")
    average = metric_average(params, default="macro")
    return BuiltinStreamingMetric(
        metric="f1",
        output_key=output_key_for("f1", average),
        average=average,
        positive_label=positive_label(params),
    )


@register_metric("f1_macro", kind="classification")
def build_f1_macro(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="f1", output_key="f1_macro", average="macro")


@register_metric("f1_micro", kind="classification")
def build_f1_micro(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="f1", output_key="f1_micro", average="micro")


@register_metric("f1_weighted", kind="classification")
def build_f1_weighted(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="f1", output_key="f1_weighted", average="weighted")
