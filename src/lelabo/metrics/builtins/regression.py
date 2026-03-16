"""Regression metrics used by supervised tasks."""

from __future__ import annotations

from ..helpers import BuiltinStreamingMetric
from ..registry import MetricContext, register_metric


@register_metric("mse", kind="regression")
def build_mse(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="mse", output_key="mse")


@register_metric("mae", kind="regression")
def build_mae(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="mae", output_key="mae")


@register_metric("rmse", kind="regression")
def build_rmse(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="rmse", output_key="rmse")


@register_metric("r2", kind="regression")
def build_r2(ctx: MetricContext):
    _ = ctx
    return BuiltinStreamingMetric(metric="r2", output_key="r2")
