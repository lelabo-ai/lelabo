from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from .base import TrainingMetric
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


class _StreamingMetricBase(TrainingMetric, ABC):
    def __init__(self, *, output_key: str):
        self.output_key = str(output_key)
        self._train_state = self.make_state()
        self._eval_state = self.make_state()
        self._last_train: float | None = None

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

    def on_epoch_start(self, trainer, epoch: int, state: Any | None = None) -> None:
        self._train_state = self.make_state()

    def on_batch_end(
        self,
        trainer,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        self.update_state(self._train_state, stats=stats, batch_size=batch_size)

    def on_epoch_end(self, trainer, epoch: int, state: Any | None = None) -> dict[str, float]:
        value = self.compute_from_state(self._train_state)
        if value is None:
            return {}
        self._last_train = float(value)
        return {self.output_key: float(self._last_train)}

    def on_train_end(self, trainer, state: Any | None = None) -> dict[str, float]:
        if self._last_train is None:
            return {}
        return {self.output_key: float(self._last_train)}

    def on_eval_start(self, trainer, split: str | None, state: Any | None = None) -> None:
        self._eval_state = self.make_state()

    def on_eval_batch_end(
        self,
        trainer,
        split: str | None,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        self.update_state(self._eval_state, stats=stats, batch_size=batch_size)

    def on_eval_end(
        self,
        trainer,
        split: str | None,
        state: Any | None = None,
    ) -> dict[str, float]:
        value = self.compute_from_state(self._eval_state)
        if value is None:
            return {}
        return {self.output_key: float(value)}


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
