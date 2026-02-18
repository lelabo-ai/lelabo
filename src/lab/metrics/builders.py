from __future__ import annotations

from .probes import BPAlignmentEpochMetric
from .registry import MetricContext, register_metric


@register_metric("bp_sign_match_epoch")
def build_bp_sign_match_epoch(_ctx: MetricContext):
    return BPAlignmentEpochMetric(
        name="bp_sign_match_epoch",
        align_prefix="bp_sign_match",
        batch_stat_key="bp_sign_match_batch",
        invert=False,
    )


@register_metric("bp_sign_mismatch_epoch")
def build_bp_sign_mismatch_epoch(_ctx: MetricContext):
    return BPAlignmentEpochMetric(
        name="bp_sign_mismatch_epoch",
        align_prefix="bp_sign_match",
        batch_stat_key="bp_sign_match_batch",
        invert=True,
    )


@register_metric("bp_cosine_epoch")
def build_bp_cosine_epoch(_ctx: MetricContext):
    return BPAlignmentEpochMetric(
        name="bp_cosine_epoch",
        align_prefix="bp_cosine",
        batch_stat_key="bp_cosine_batch",
        invert=False,
    )


@register_metric("bp_update_gap_epoch")
def build_bp_update_gap_epoch(_ctx: MetricContext):
    return BPAlignmentEpochMetric(
        name="bp_update_gap_epoch",
        align_prefix="bp_update_gap",
        batch_stat_key="bp_update_gap_batch",
        invert=False,
    )
