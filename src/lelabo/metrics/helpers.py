from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from .base import TrainerMetric
from .payload import extract_metric_payload


@dataclass
class _ScalarState:
    value_sum: float = 0.0
    weight_sum: float = 0.0

    def update(self, value: float, weight: float) -> None:
        self.value_sum += float(value) * float(weight)
        self.weight_sum += float(weight)


@dataclass
class _RegressionState:
    sse: float = 0.0
    sae: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    count: float = 0.0

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        true_f = y_true.reshape(-1).to(dtype=torch.float64)
        pred_f = y_pred.reshape(-1).to(dtype=torch.float64)
        if true_f.numel() == 0 or pred_f.numel() == 0:
            return
        n = int(min(true_f.numel(), pred_f.numel()))
        true_f = true_f[:n]
        pred_f = pred_f[:n]
        err = pred_f - true_f
        self.sse += float((err * err).sum().item())
        self.sae += float(err.abs().sum().item())
        self.sum_y += float(true_f.sum().item())
        self.sum_y2 += float((true_f * true_f).sum().item())
        self.count += float(n)


class _ClassificationState:
    def __init__(self) -> None:
        self.cm = torch.zeros((0, 0), dtype=torch.float64)

    def _ensure_size(self, size: int) -> None:
        if size <= int(self.cm.size(0)):
            return
        old = self.cm
        self.cm = torch.zeros((size, size), dtype=torch.float64)
        if old.numel() > 0:
            n = int(old.size(0))
            self.cm[:n, :n] = old

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        true_i = y_true.reshape(-1).to(dtype=torch.int64)
        pred_i = y_pred.reshape(-1).to(dtype=torch.int64)
        if true_i.numel() == 0 or pred_i.numel() == 0:
            return
        n = int(min(true_i.numel(), pred_i.numel()))
        true_i = true_i[:n]
        pred_i = pred_i[:n]
        valid = (true_i >= 0) & (pred_i >= 0)
        if not bool(valid.any()):
            return
        true_i = true_i[valid]
        pred_i = pred_i[valid]
        size = int(max(int(true_i.max().item()), int(pred_i.max().item())) + 1)
        self._ensure_size(size)
        k = int(self.cm.size(0))
        idx = true_i * k + pred_i
        bincount = torch.bincount(idx, minlength=k * k).reshape(k, k).to(dtype=torch.float64)
        self.cm += bincount


class _StreamingMetricBase(TrainerMetric, ABC):
    def __init__(self, *, output_key: str):
        self.output_key = str(output_key)
        self._state_by_split: dict[str, Any] = {}
        self._last_by_split: dict[str, dict[str, float]] = {}

    @staticmethod
    def _normalize_split(split: str | None) -> str:
        token = str(split or "").strip().lower()
        return token or "eval"

    @abstractmethod
    def make_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        state: Any,
        *,
        stats: dict[str, Any],
        batch_size: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_from_state(self, state: Any) -> float | None:
        raise NotImplementedError

    def reset(self, split: str, state: Any | None = None) -> None:
        _ = state
        self._state_by_split[self._normalize_split(split)] = self.make_state()

    def update(
        self,
        split: str,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        _ = state
        split_key = self._normalize_split(split)
        if split_key not in self._state_by_split:
            self._state_by_split[split_key] = self.make_state()
        self.update_state(self._state_by_split[split_key], stats=stats, batch_size=batch_size)

    def compute(self, split: str, state: Any | None = None) -> dict[str, float]:
        _ = state
        split_key = self._normalize_split(split)
        if split_key not in self._state_by_split:
            return {}
        value = self.compute_from_state(self._state_by_split[split_key])
        if value is None:
            return {}
        out = {self.output_key: float(value)}
        self._last_by_split[split_key] = dict(out)
        return out

    def finalize(self, state: Any | None = None) -> dict[str, float]:
        _ = state
        return dict(self._last_by_split.get("train", {}))


class ClassificationStreamingMetric(_StreamingMetricBase, ABC):
    """Streaming classification metric backed by a confusion matrix."""

    def make_state(self) -> _ClassificationState:
        return _ClassificationState()

    def update_state(
        self,
        state: _ClassificationState,
        *,
        stats: dict[str, Any],
        batch_size: int,
    ) -> None:
        _ = batch_size
        payload = extract_metric_payload(stats)
        if payload is None or payload.kind != "classification":
            return
        state.update(payload.y_true, payload.y_pred)

    def compute_from_state(self, state: _ClassificationState) -> float | None:
        return self.compute_from_confusion_matrix(state.cm.clone())

    @abstractmethod
    def compute_from_confusion_matrix(self, confusion_matrix: torch.Tensor) -> float | None:
        raise NotImplementedError


class RegressionStreamingMetric(_StreamingMetricBase, ABC):
    """Streaming regression metric backed by running error summaries."""

    def make_state(self) -> _RegressionState:
        return _RegressionState()

    def update_state(
        self,
        state: _RegressionState,
        *,
        stats: dict[str, Any],
        batch_size: int,
    ) -> None:
        _ = batch_size
        payload = extract_metric_payload(stats)
        if payload is None or payload.kind != "regression":
            return
        state.update(payload.y_true, payload.y_pred)

    def compute_from_state(self, state: _RegressionState) -> float | None:
        return self.compute_from_regression_state(
            sse=float(state.sse),
            sae=float(state.sae),
            sum_y=float(state.sum_y),
            sum_y2=float(state.sum_y2),
            count=float(state.count),
        )

    @abstractmethod
    def compute_from_regression_state(
        self,
        *,
        sse: float,
        sae: float,
        sum_y: float,
        sum_y2: float,
        count: float,
    ) -> float | None:
        raise NotImplementedError


class ScalarStreamingMetric(_StreamingMetricBase, ABC):
    """Streaming scalar metric backed by weighted running sums."""

    def __init__(self, *, stat_key: str, output_key: str):
        super().__init__(output_key=output_key)
        self.stat_key = str(stat_key)

    def make_state(self) -> _ScalarState:
        return _ScalarState()

    def extract_scalar_value(self, stats: dict[str, Any]) -> float | None:
        value = stats.get(self.stat_key)
        if not isinstance(value, (int, float)):
            return None
        return float(value)

    def update_state(
        self,
        state: _ScalarState,
        *,
        stats: dict[str, Any],
        batch_size: int,
    ) -> None:
        value = self.extract_scalar_value(stats)
        if value is None:
            return
        state.update(value, float(max(1, int(batch_size))))

    def compute_from_state(self, state: _ScalarState) -> float | None:
        return self.compute_from_scalar_state(
            value_sum=float(state.value_sum),
            weight_sum=float(state.weight_sum),
        )

    @abstractmethod
    def compute_from_scalar_state(self, *, value_sum: float, weight_sum: float) -> float | None:
        raise NotImplementedError


class ScalarMeanMetric(ScalarStreamingMetric):
    """Utility metric that averages a scalar key over train/eval batches."""

    def __init__(self, *, stat_key: str, output_key: str):
        super().__init__(stat_key=stat_key, output_key=output_key)

    def compute_from_scalar_state(self, *, value_sum: float, weight_sum: float) -> float | None:
        if weight_sum <= 0.0:
            return None
        return float(value_sum / weight_sum)


class ClassificationMetricBase(ClassificationStreamingMetric):
    """Backward-compatible alias for the streaming classification base."""


class RegressionMetricBase(RegressionStreamingMetric):
    """Backward-compatible alias for the streaming regression base."""


def _safe_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not torch.isfinite(torch.tensor(out)):
        return None
    return out


@dataclass
class _BuiltinMetricState:
    scalar: _ScalarState
    cls: _ClassificationState
    reg: _RegressionState


def _new_builtin_state() -> _BuiltinMetricState:
    return _BuiltinMetricState(
        scalar=_ScalarState(),
        cls=_ClassificationState(),
        reg=_RegressionState(),
    )


class BuiltinStreamingMetric(_StreamingMetricBase):
    def __init__(
        self,
        *,
        metric: str,
        output_key: str,
        average: str = "macro",
        positive_label: int = 1,
    ):
        super().__init__(output_key=output_key)
        self.metric = str(metric).strip().lower()
        self.average = str(average).strip().lower()
        self.positive_label = int(positive_label)

    def make_state(self) -> _BuiltinMetricState:
        return _new_builtin_state()

    def update_state(
        self,
        state: _BuiltinMetricState,
        *,
        stats: dict[str, Any],
        batch_size: int,
    ) -> None:
        payload = extract_metric_payload(stats)
        if payload is not None:
            if self.metric in {"accuracy", "precision", "recall", "f1"} and payload.kind == "classification":
                state.cls.update(payload.y_true, payload.y_pred)
                return
            if self.metric in {"mse", "mae", "rmse", "r2"} and payload.kind == "regression":
                state.reg.update(payload.y_true, payload.y_pred)
                return

        fallback = self._fallback_value(stats)
        if fallback is None:
            return
        state.scalar.update(fallback, float(max(1, int(batch_size))))

    def _fallback_value(self, stats: dict[str, Any]) -> float | None:
        keys: list[str]
        if self.metric == "accuracy":
            keys = [self.output_key, "acc", "accuracy"]
        elif self.metric == "precision":
            keys = [self.output_key, "precision"]
        elif self.metric == "recall":
            keys = [self.output_key, "recall"]
        elif self.metric == "f1":
            keys = [self.output_key, "f1"]
        elif self.metric in {"mse", "mae", "rmse", "r2"}:
            keys = [self.output_key, self.metric]
        else:
            return None
        for key in keys:
            value = _safe_float(stats.get(key))
            if value is not None:
                return value
        return None

    def compute_from_state(self, state: _BuiltinMetricState) -> float | None:
        if self.metric == "accuracy":
            return _coalesce(_classification_accuracy(state.cls.cm), _scalar_value(state.scalar))
        if self.metric == "precision":
            return _coalesce(
                _classification_precision(state.cls.cm, average=self.average, positive_label=self.positive_label),
                _scalar_value(state.scalar),
            )
        if self.metric == "recall":
            return _coalesce(
                _classification_recall(state.cls.cm, average=self.average, positive_label=self.positive_label),
                _scalar_value(state.scalar),
            )
        if self.metric == "f1":
            return _coalesce(
                _classification_f1(state.cls.cm, average=self.average, positive_label=self.positive_label),
                _scalar_value(state.scalar),
            )
        if self.metric == "mse":
            return _coalesce(_regression_mse(state.reg), _scalar_value(state.scalar))
        if self.metric == "mae":
            return _coalesce(_regression_mae(state.reg), _scalar_value(state.scalar))
        if self.metric == "rmse":
            return _coalesce(_regression_rmse(state.reg), _scalar_value(state.scalar))
        if self.metric == "r2":
            return _coalesce(_regression_r2(state.reg), _scalar_value(state.scalar))
        return None


def _scalar_value(state: _ScalarState) -> float | None:
    if state.weight_sum <= 0.0:
        return None
    return float(state.value_sum / state.weight_sum)


def _classification_total(cm: torch.Tensor) -> float:
    if cm.numel() == 0:
        return 0.0
    return float(cm.sum().item())


def _classification_accuracy(cm: torch.Tensor) -> float | None:
    total = _classification_total(cm)
    if total <= 0.0:
        return None
    return float(cm.diag().sum().item() / total)


def _classification_stats(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    support = cm.sum(dim=1)
    return tp, fp, fn, support


def _average_by_mode(values: torch.Tensor, *, support: torch.Tensor, average: str) -> float | None:
    if values.numel() == 0:
        return None
    valid = support > 0
    if not bool(valid.any()):
        return 0.0
    if average == "weighted":
        weights = support[valid]
        denom = float(weights.sum().item())
        if denom <= 0.0:
            return 0.0
        return float((values[valid] * weights).sum().item() / denom)
    return float(values[valid].mean().item())


def _classification_precision(cm: torch.Tensor, *, average: str, positive_label: int) -> float | None:
    tp, fp, _, support = _classification_stats(cm)
    if tp.numel() == 0:
        return None
    if average == "micro":
        den = float((tp + fp).sum().item())
        if den <= 0.0:
            return 0.0
        return float(tp.sum().item() / den)
    if average == "binary":
        idx = int(max(0, min(int(positive_label), int(tp.numel() - 1))))
        den = float((tp[idx] + fp[idx]).item())
        if den <= 0.0:
            return 0.0
        return float(tp[idx].item() / den)
    per_class = torch.where((tp + fp) > 0.0, tp / (tp + fp), torch.zeros_like(tp))
    return _average_by_mode(per_class, support=support, average=average)


def _classification_recall(cm: torch.Tensor, *, average: str, positive_label: int) -> float | None:
    tp, _, fn, support = _classification_stats(cm)
    if tp.numel() == 0:
        return None
    if average == "micro":
        den = float((tp + fn).sum().item())
        if den <= 0.0:
            return 0.0
        return float(tp.sum().item() / den)
    if average == "binary":
        idx = int(max(0, min(int(positive_label), int(tp.numel() - 1))))
        den = float((tp[idx] + fn[idx]).item())
        if den <= 0.0:
            return 0.0
        return float(tp[idx].item() / den)
    per_class = torch.where((tp + fn) > 0.0, tp / (tp + fn), torch.zeros_like(tp))
    return _average_by_mode(per_class, support=support, average=average)


def _classification_f1(cm: torch.Tensor, *, average: str, positive_label: int) -> float | None:
    tp, fp, fn, support = _classification_stats(cm)
    if tp.numel() == 0:
        return None
    if average == "micro":
        den_p = float((tp + fp).sum().item())
        den_r = float((tp + fn).sum().item())
        if den_p <= 0.0 or den_r <= 0.0:
            return 0.0
        precision = float(tp.sum().item() / den_p)
        recall = float(tp.sum().item() / den_r)
        den = precision + recall
        if den <= 0.0:
            return 0.0
        return float((2.0 * precision * recall) / den)
    if average == "binary":
        idx = int(max(0, min(int(positive_label), int(tp.numel() - 1))))
        p_den = float((tp[idx] + fp[idx]).item())
        r_den = float((tp[idx] + fn[idx]).item())
        precision = 0.0 if p_den <= 0.0 else float(tp[idx].item() / p_den)
        recall = 0.0 if r_den <= 0.0 else float(tp[idx].item() / r_den)
        den = precision + recall
        if den <= 0.0:
            return 0.0
        return float((2.0 * precision * recall) / den)
    precision = torch.where((tp + fp) > 0.0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where((tp + fn) > 0.0, tp / (tp + fn), torch.zeros_like(tp))
    den = precision + recall
    f1 = torch.where(den > 0.0, (2.0 * precision * recall) / den, torch.zeros_like(den))
    return _average_by_mode(f1, support=support, average=average)


def _regression_mse(state: _RegressionState) -> float | None:
    if state.count <= 0.0:
        return None
    return float(state.sse / state.count)


def _regression_mae(state: _RegressionState) -> float | None:
    if state.count <= 0.0:
        return None
    return float(state.sae / state.count)


def _regression_rmse(state: _RegressionState) -> float | None:
    mse = _regression_mse(state)
    if mse is None:
        return None
    return float(torch.sqrt(torch.tensor(mse)).item())


def _regression_r2(state: _RegressionState) -> float | None:
    if state.count <= 0.0:
        return None
    mean_y = state.sum_y / state.count
    sst = state.sum_y2 - state.count * mean_y * mean_y
    if sst <= 0.0:
        return 0.0
    return float(1.0 - (state.sse / sst))


def _coalesce(primary: float | None, secondary: float | None) -> float | None:
    if primary is not None:
        return primary
    return secondary
