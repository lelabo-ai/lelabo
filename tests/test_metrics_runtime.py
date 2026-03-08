from __future__ import annotations

import importlib
import math
import sys
import uuid
from argparse import Namespace

import pytest
import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
metrics_api = importlib.import_module("lelabo.metrics")
metrics_payload = importlib.import_module("lelabo.metrics.payload")
metrics_registry = importlib.import_module("lelabo.metrics.registry")


@pytest.fixture(autouse=True)
def _restore_metric_registry_state():
    original_items = dict(metrics_registry.METRIC_REGISTRY._items)
    original_base = metrics_registry._BASE_METRIC_ITEMS
    original_last_key = metrics_registry._LAST_METRIC_REFRESH_KEY
    original_last_items = metrics_registry._LAST_METRIC_ITEMS
    try:
        yield
    finally:
        metrics_registry.METRIC_REGISTRY._items = dict(original_items)
        metrics_registry._BASE_METRIC_ITEMS = original_base
        metrics_registry._LAST_METRIC_REFRESH_KEY = original_last_key
        metrics_registry._LAST_METRIC_ITEMS = original_last_items


def _ctx(*, extra: dict | None = None):
    return metrics_api.MetricContext(
        args=Namespace(),
        mode="supervised",
        dataset="iris",
        algo="bp",
        extra=dict(extra or {}),
    )


def _new_metric_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def test_parse_metric_names_normalizes_and_deduplicates() -> None:
    out = metrics_api.parse_metric_names(" f1, ACC  f1,  RMSE  acc ")
    assert out == ["f1", "acc", "rmse"]


def test_parse_metric_names_supports_iterables() -> None:
    out = metrics_api.parse_metric_names(["f1,acc", " rmse ", "f1"])
    assert out == ["f1", "acc", "rmse"]


def test_validate_metric_requests_rejects_invalid_task_kind() -> None:
    with pytest.raises(ValueError, match="task_kind must be one of"):
        metrics_api.validate_metric_requests(["f1"], ctx=_ctx(), task_kind="scalar")


def test_validate_metric_requests_rejects_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="Metric 'mse' is for regression, but task is classification"):
        metrics_api.validate_metric_requests(["mse"], ctx=_ctx(), task_kind="classification")


def test_validate_metric_requests_rejects_unknown_metric_params() -> None:
    ctx = _ctx(extra={"metric_params": {"f1": {"average": "macro", "unknown_knob": 1}}})
    with pytest.raises(ValueError, match="Unknown params for metric 'f1'"):
        metrics_api.validate_metric_requests(["f1"], ctx=ctx, task_kind="classification")


def test_validate_metric_requests_ignores_unknown_metric_name() -> None:
    metrics_api.validate_metric_requests(["does_not_exist"], ctx=_ctx(), task_kind="classification")


@pytest.mark.parametrize(
    ("metric_name", "expected_key", "expected_value"),
    [
        ("mse", "mse", 8.0 / 3.0),
        ("mae", "mae", 4.0 / 3.0),
        ("rmse", "rmse", math.sqrt(8.0 / 3.0)),
        ("r2", "r2", -3.0),
    ],
)
def test_builtin_regression_metrics_compute_from_payload(
    metric_name: str,
    expected_key: str,
    expected_value: float,
) -> None:
    probe = metrics_api.build_metric(metric_name, _ctx())
    probe.on_epoch_start(None, 1, None)
    probe.on_batch_end(
        None,
        stats={
            metrics_payload.METRIC_Y_TRUE_KEY: torch.tensor([1.0, 2.0, 3.0]),
            metrics_payload.METRIC_Y_PRED_KEY: torch.tensor([1.0, 4.0, 1.0]),
            metrics_payload.METRIC_KIND_KEY: "regression",
        },
        batch_size=3,
        state=None,
    )
    out = probe.on_epoch_end(None, 1, None)
    assert expected_key in out
    assert out[expected_key] == pytest.approx(expected_value, rel=1e-6)


def test_metrics_api_no_longer_exports_register_metric_fn() -> None:
    assert not hasattr(metrics_api, "register_metric_fn")


def test_register_streaming_classification_metric_smoke() -> None:
    name = _new_metric_name("unit_cls_metric")
    output_key = f"{name}_score"

    class _ExactMatchRate(metrics_api.ClassificationStreamingMetric):
        def compute_from_confusion_matrix(self, confusion_matrix: torch.Tensor) -> float | None:
            total = float(confusion_matrix.sum().item())
            if total <= 0.0:
                return None
            return float(confusion_matrix.diag().sum().item() / total)

    @metrics_api.register_metric(
        name,
        kind="classification",
        params={"bonus": "float additive offset"},
    )
    def _builder(ctx):
        params = ctx.metric_params(name)
        bonus = float(params.get("bonus", 0.0))

        class _MetricWithBonus(_ExactMatchRate):
            def compute_from_confusion_matrix(self, confusion_matrix: torch.Tensor) -> float | None:
                base = super().compute_from_confusion_matrix(confusion_matrix)
                if base is None:
                    return None
                return float(base + bonus)

        return _MetricWithBonus(output_key=output_key)

    builder = metrics_registry.METRIC_REGISTRY.get(name)
    assert getattr(builder, "__metric_kind__", None) == "classification"
    assert getattr(builder, "__metric_params__", None) == ("bonus",)

    probe = builder(_ctx(extra={"metric_params": {name: {"bonus": 0.25}}}))
    probe.on_epoch_start(None, 1, None)
    probe.on_batch_end(
        None,
        stats={
            metrics_payload.METRIC_Y_TRUE_KEY: torch.tensor([0, 1, 1, 0]),
            metrics_payload.METRIC_Y_PRED_KEY: torch.tensor([0, 1, 0, 0]),
            metrics_payload.METRIC_KIND_KEY: "classification",
        },
        batch_size=4,
        state=None,
    )
    out = probe.on_epoch_end(None, 1, None)
    assert output_key in out
    assert out[output_key] == pytest.approx(1.0, rel=1e-6)
    assert hasattr(probe, "_train_state")
    assert not isinstance(probe._train_state, list)


def test_register_streaming_regression_metric_smoke() -> None:
    name = _new_metric_name("unit_reg_metric")
    output_key = f"{name}_score"

    class _ShiftedMae(metrics_api.RegressionStreamingMetric):
        def __init__(self, *, shift: float):
            super().__init__(output_key=output_key)
            self.shift = float(shift)

        def compute_from_regression_state(
            self,
            *,
            sse: float,
            sae: float,
            sum_y: float,
            sum_y2: float,
            count: float,
        ) -> float | None:
            _ = (sse, sum_y, sum_y2)
            if count <= 0.0:
                return None
            return float((sae / count) + self.shift)

    @metrics_api.register_metric(name, kind="regression", params={"shift": "float additive offset"})
    def _builder(ctx):
        params = ctx.metric_params(name)
        return _ShiftedMae(shift=float(params.get("shift", 0.0)))

    builder = metrics_registry.METRIC_REGISTRY.get(name)
    assert getattr(builder, "__metric_kind__", None) == "regression"

    probe = builder(_ctx(extra={"metric_params": {name: {"shift": 0.5}}}))
    probe.on_epoch_start(None, 1, None)
    probe.on_batch_end(
        None,
        stats={
            metrics_payload.METRIC_Y_TRUE_KEY: torch.tensor([1.0, 2.0]),
            metrics_payload.METRIC_Y_PRED_KEY: torch.tensor([1.0, 5.0]),
            metrics_payload.METRIC_KIND_KEY: "regression",
        },
        batch_size=2,
        state=None,
    )
    out = probe.on_epoch_end(None, 1, None)
    assert output_key in out
    assert out[output_key] == pytest.approx(2.0, rel=1e-6)
    assert hasattr(probe, "_train_state")
    assert not isinstance(probe._train_state, list)


def test_register_streaming_scalar_metric_smoke() -> None:
    name = _new_metric_name("unit_scalar_metric")
    output_key = f"{name}_score"

    class _ScaledMean(metrics_api.ScalarStreamingMetric):
        def __init__(self, *, scale: float):
            super().__init__(stat_key="my_loss", output_key=output_key)
            self.scale = float(scale)

        def compute_from_scalar_state(self, *, value_sum: float, weight_sum: float) -> float | None:
            if weight_sum <= 0.0:
                return None
            return float((value_sum / weight_sum) * self.scale)

    @metrics_api.register_metric(name, kind="scalar", params={"scale": "multiply the running mean"})
    def _builder(ctx):
        params = ctx.metric_params(name)
        return _ScaledMean(scale=float(params.get("scale", 1.0)))

    builder = metrics_registry.METRIC_REGISTRY.get(name)
    assert getattr(builder, "__metric_kind__", None) == "scalar"

    probe = builder(_ctx(extra={"metric_params": {name: {"scale": 2.0}}}))
    probe.on_epoch_start(None, 1, None)
    probe.on_batch_end(None, stats={"my_loss": 1.0}, batch_size=2, state=None)
    probe.on_batch_end(None, stats={"my_loss": 3.0}, batch_size=1, state=None)
    out = probe.on_epoch_end(None, 1, None)
    assert output_key in out
    assert out[output_key] == pytest.approx((5.0 / 3.0) * 2.0, rel=1e-6)
    assert hasattr(probe, "_train_state")
    assert not isinstance(probe._train_state, list)


def test_streaming_metrics_keep_split_states_separate() -> None:
    class _Accuracy(metrics_api.ClassificationStreamingMetric):
        def __init__(self):
            super().__init__(output_key="stream_acc")

        def compute_from_confusion_matrix(self, confusion_matrix: torch.Tensor) -> float | None:
            total = float(confusion_matrix.sum().item())
            if total <= 0.0:
                return None
            return float(confusion_matrix.diag().sum().item() / total)

    probe = _Accuracy()
    probe.on_epoch_start(None, 1, None)
    probe.on_batch_end(
        None,
        stats={
            metrics_payload.METRIC_Y_TRUE_KEY: torch.tensor([0, 1]),
            metrics_payload.METRIC_Y_PRED_KEY: torch.tensor([0, 1]),
            metrics_payload.METRIC_KIND_KEY: "classification",
        },
        batch_size=2,
        state=None,
    )
    train_out = probe.on_epoch_end(None, 1, None)

    probe.on_eval_start(None, "val", None)
    probe.on_eval_batch_end(
        None,
        "val",
        stats={
            metrics_payload.METRIC_Y_TRUE_KEY: torch.tensor([0, 1]),
            metrics_payload.METRIC_Y_PRED_KEY: torch.tensor([1, 1]),
            metrics_payload.METRIC_KIND_KEY: "classification",
        },
        batch_size=2,
        state=None,
    )
    eval_out = probe.on_eval_end(None, "val", None)

    assert train_out["stream_acc"] == pytest.approx(1.0, rel=1e-6)
    assert eval_out["stream_acc"] == pytest.approx(0.5, rel=1e-6)
    assert probe.on_train_end(None, None)["stream_acc"] == pytest.approx(1.0, rel=1e-6)


def test_register_metric_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported metric kind"):
        @metrics_api.register_metric(_new_metric_name("invalid_metric"), kind="unsupported_kind")
        def _builder(_ctx):  # pragma: no cover - should never execute
            return object()


def test_metric_payload_roundtrip_and_validation() -> None:
    y_true = torch.tensor([0.0, 1.0], requires_grad=True)
    y_pred = torch.tensor([0.0, 0.0], requires_grad=True)
    payload = metrics_payload.build_metric_payload(
        y_true=y_true,
        y_pred=y_pred,
        kind="classification",
    )
    assert set(payload.keys()) == {
        metrics_payload.METRIC_Y_TRUE_KEY,
        metrics_payload.METRIC_Y_PRED_KEY,
        metrics_payload.METRIC_KIND_KEY,
    }
    assert payload[metrics_payload.METRIC_Y_TRUE_KEY].requires_grad is False
    assert payload[metrics_payload.METRIC_Y_PRED_KEY].requires_grad is False
    extracted = metrics_payload.extract_metric_payload(payload)
    assert extracted is not None
    assert extracted.kind == "classification"
    assert torch.equal(extracted.y_true, torch.tensor([0.0, 1.0]))
    assert torch.equal(extracted.y_pred, torch.tensor([0.0, 0.0]))

    assert metrics_payload.extract_metric_payload({metrics_payload.METRIC_KIND_KEY: "classification"}) is None
    invalid_kind_payload = dict(payload)
    invalid_kind_payload[metrics_payload.METRIC_KIND_KEY] = "scalar"
    assert metrics_payload.extract_metric_payload(invalid_kind_payload) is None
