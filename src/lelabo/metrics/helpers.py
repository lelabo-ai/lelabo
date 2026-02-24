from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from .base import TrainingMetric
from .payload import extract_metric_payload


class ScalarMeanMetric(TrainingMetric):
    """Utility metric that averages a scalar key over train/eval batches."""

    def __init__(self, *, stat_key: str, output_key: str):
        self.stat_key = str(stat_key)
        self.output_key = str(output_key)
        self._train_sum = 0.0
        self._train_count = 0.0
        self._eval_sum = 0.0
        self._eval_count = 0.0
        self._last_train: float | None = None

    def on_epoch_start(self, trainer, epoch: int, state: Any | None = None) -> None:
        self._train_sum = 0.0
        self._train_count = 0.0

    def on_batch_end(
        self,
        trainer,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        value = stats.get(self.stat_key)
        if not isinstance(value, (int, float)):
            return
        w = float(max(1, int(batch_size)))
        self._train_sum += float(value) * w
        self._train_count += w

    def on_epoch_end(self, trainer, epoch: int, state: Any | None = None) -> dict[str, float]:
        if self._train_count <= 0.0:
            return {}
        self._last_train = float(self._train_sum / self._train_count)
        return {self.output_key: float(self._last_train)}

    def on_train_end(self, trainer, state: Any | None = None) -> dict[str, float]:
        if self._last_train is None:
            return {}
        return {self.output_key: float(self._last_train)}

    def on_eval_start(self, trainer, split: str | None, state: Any | None = None) -> None:
        self._eval_sum = 0.0
        self._eval_count = 0.0

    def on_eval_batch_end(
        self,
        trainer,
        split: str | None,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        value = stats.get(self.stat_key)
        if not isinstance(value, (int, float)):
            return
        w = float(max(1, int(batch_size)))
        self._eval_sum += float(value) * w
        self._eval_count += w

    def on_eval_end(
        self,
        trainer,
        split: str | None,
        state: Any | None = None,
    ) -> dict[str, float]:
        if self._eval_count <= 0.0:
            return {}
        return {self.output_key: float(self._eval_sum / self._eval_count)}


class _PayloadMetricBase(TrainingMetric, ABC):
    def __init__(self, *, output_key: str, expected_kind: str):
        self.output_key = str(output_key)
        self.expected_kind = str(expected_kind).strip().lower()
        self._train_true: list[torch.Tensor] = []
        self._train_pred: list[torch.Tensor] = []
        self._eval_true: list[torch.Tensor] = []
        self._eval_pred: list[torch.Tensor] = []
        self._last_train: float | None = None

    def on_epoch_start(self, trainer, epoch: int, state: Any | None = None) -> None:
        self._train_true = []
        self._train_pred = []

    def on_batch_end(
        self,
        trainer,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        _ = batch_size
        payload = extract_metric_payload(stats)
        if payload is None or payload.kind != self.expected_kind:
            return
        self._train_true.append(payload.y_true)
        self._train_pred.append(payload.y_pred)

    def on_epoch_end(self, trainer, epoch: int, state: Any | None = None) -> dict[str, float]:
        value = self._compute(self._train_true, self._train_pred)
        if value is None:
            return {}
        self._last_train = float(value)
        return {self.output_key: float(self._last_train)}

    def on_train_end(self, trainer, state: Any | None = None) -> dict[str, float]:
        if self._last_train is None:
            return {}
        return {self.output_key: float(self._last_train)}

    def on_eval_start(self, trainer, split: str | None, state: Any | None = None) -> None:
        self._eval_true = []
        self._eval_pred = []

    def on_eval_batch_end(
        self,
        trainer,
        split: str | None,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        _ = split, batch_size, state
        payload = extract_metric_payload(stats)
        if payload is None or payload.kind != self.expected_kind:
            return
        self._eval_true.append(payload.y_true)
        self._eval_pred.append(payload.y_pred)

    def on_eval_end(
        self,
        trainer,
        split: str | None,
        state: Any | None = None,
    ) -> dict[str, float]:
        _ = split, state
        value = self._compute(self._eval_true, self._eval_pred)
        if value is None:
            return {}
        return {self.output_key: float(value)}

    @abstractmethod
    def compute_from_tensors(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float | None:
        raise NotImplementedError

    def _compute(self, y_true_parts: list[torch.Tensor], y_pred_parts: list[torch.Tensor]) -> float | None:
        if not y_true_parts or not y_pred_parts:
            return None
        y_true = torch.cat([x.reshape(-1) for x in y_true_parts], dim=0)
        y_pred = torch.cat([x.reshape(-1) for x in y_pred_parts], dim=0)
        n = int(min(y_true.numel(), y_pred.numel()))
        if n <= 0:
            return None
        return self.compute_from_tensors(y_true[:n], y_pred[:n])


class ClassificationMetricBase(_PayloadMetricBase):
    """Utility base class for classification metrics driven by y_true/y_pred payload."""

    def __init__(self, *, output_key: str):
        super().__init__(output_key=output_key, expected_kind="classification")


class RegressionMetricBase(_PayloadMetricBase):
    """Utility base class for regression metrics driven by y_true/y_pred payload."""

    def __init__(self, *, output_key: str):
        super().__init__(output_key=output_key, expected_kind="regression")


class FunctionClassificationMetric(ClassificationMetricBase):
    def __init__(
        self,
        *,
        output_key: str,
        fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], float],
        metric_params: dict[str, Any] | None = None,
    ):
        super().__init__(output_key=output_key)
        self.fn = fn
        self.metric_params = dict(metric_params or {})

    def compute_from_tensors(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float | None:
        value = self.fn(y_true, y_pred, self.metric_params)
        if not isinstance(value, (int, float)):
            return None
        return float(value)


class FunctionRegressionMetric(RegressionMetricBase):
    def __init__(
        self,
        *,
        output_key: str,
        fn: Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], float],
        metric_params: dict[str, Any] | None = None,
    ):
        super().__init__(output_key=output_key)
        self.fn = fn
        self.metric_params = dict(metric_params or {})

    def compute_from_tensors(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float | None:
        value = self.fn(y_true, y_pred, self.metric_params)
        if not isinstance(value, (int, float)):
            return None
        return float(value)


class FunctionScalarMetric(TrainingMetric):
    """Function metric for scalar streams (value_sum, weight_sum, params) -> float."""

    def __init__(
        self,
        *,
        output_key: str,
        fn: Callable[[float, float, dict[str, Any]], float],
        metric_params: dict[str, Any] | None = None,
    ):
        self.output_key = str(output_key)
        self.fn = fn
        self.metric_params = dict(metric_params or {})
        self._train_sum = 0.0
        self._train_count = 0.0
        self._eval_sum = 0.0
        self._eval_count = 0.0
        self._last_train: float | None = None

    def _scalar_key(self) -> str:
        key = self.metric_params.get("key", "loss")
        return str(key)

    def on_epoch_start(self, trainer, epoch: int, state: Any | None = None) -> None:
        self._train_sum = 0.0
        self._train_count = 0.0

    def on_batch_end(
        self,
        trainer,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        value = stats.get(self._scalar_key())
        if not isinstance(value, (int, float)):
            return
        w = float(max(1, int(batch_size)))
        self._train_sum += float(value) * w
        self._train_count += w

    def on_epoch_end(self, trainer, epoch: int, state: Any | None = None) -> dict[str, float]:
        if self._train_count <= 0.0:
            return {}
        out = self.fn(float(self._train_sum), float(self._train_count), dict(self.metric_params))
        if not isinstance(out, (int, float)):
            return {}
        self._last_train = float(out)
        return {self.output_key: float(self._last_train)}

    def on_train_end(self, trainer, state: Any | None = None) -> dict[str, float]:
        if self._last_train is None:
            return {}
        return {self.output_key: float(self._last_train)}

    def on_eval_start(self, trainer, split: str | None, state: Any | None = None) -> None:
        self._eval_sum = 0.0
        self._eval_count = 0.0

    def on_eval_batch_end(
        self,
        trainer,
        split: str | None,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        value = stats.get(self._scalar_key())
        if not isinstance(value, (int, float)):
            return
        w = float(max(1, int(batch_size)))
        self._eval_sum += float(value) * w
        self._eval_count += w

    def on_eval_end(
        self,
        trainer,
        split: str | None,
        state: Any | None = None,
    ) -> dict[str, float]:
        if self._eval_count <= 0.0:
            return {}
        out = self.fn(float(self._eval_sum), float(self._eval_count), dict(self.metric_params))
        if not isinstance(out, (int, float)):
            return {}
        return {self.output_key: float(out)}
