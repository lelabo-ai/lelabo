from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .base import TrainingMetric
from .payload import extract_metric_payload


def _safe_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


class _RunningScalar:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: float, weight: float) -> None:
        self.sum += float(value) * float(weight)
        self.count += float(weight)

    def value(self) -> float | None:
        if self.count <= 0.0:
            return None
        return float(self.sum / self.count)


class _ClassificationAccumulator:
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

    def total(self) -> float:
        if self.cm.numel() == 0:
            return 0.0
        return float(self.cm.sum().item())

    def accuracy(self) -> float | None:
        total = self.total()
        if total <= 0.0:
            return None
        return float(self.cm.diag().sum().item() / total)

    def _class_stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tp = self.cm.diag()
        fp = self.cm.sum(dim=0) - tp
        fn = self.cm.sum(dim=1) - tp
        support = self.cm.sum(dim=1)
        return tp, fp, fn, support

    def precision(self, *, average: str, positive_label: int = 1) -> float | None:
        tp, fp, _, support = self._class_stats()
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

    def recall(self, *, average: str, positive_label: int = 1) -> float | None:
        tp, _, fn, support = self._class_stats()
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

    def f1(self, *, average: str, positive_label: int = 1) -> float | None:
        tp, fp, fn, support = self._class_stats()
        if tp.numel() == 0:
            return None
        if average == "micro":
            den_p = float((tp + fp).sum().item())
            den_r = float((tp + fn).sum().item())
            if den_p <= 0.0 or den_r <= 0.0:
                return 0.0
            p = float(tp.sum().item() / den_p)
            r = float(tp.sum().item() / den_r)
            den = p + r
            if den <= 0.0:
                return 0.0
            return float((2.0 * p * r) / den)
        if average == "binary":
            idx = int(max(0, min(int(positive_label), int(tp.numel() - 1))))
            p_den = float((tp[idx] + fp[idx]).item())
            r_den = float((tp[idx] + fn[idx]).item())
            p = 0.0 if p_den <= 0.0 else float(tp[idx].item() / p_den)
            r = 0.0 if r_den <= 0.0 else float(tp[idx].item() / r_den)
            den = p + r
            if den <= 0.0:
                return 0.0
            return float((2.0 * p * r) / den)
        p = torch.where((tp + fp) > 0.0, tp / (tp + fp), torch.zeros_like(tp))
        r = torch.where((tp + fn) > 0.0, tp / (tp + fn), torch.zeros_like(tp))
        den = p + r
        f1 = torch.where(den > 0.0, (2.0 * p * r) / den, torch.zeros_like(den))
        return _average_by_mode(f1, support=support, average=average)


def _average_by_mode(values: torch.Tensor, *, support: torch.Tensor, average: str) -> float | None:
    if values.numel() == 0:
        return None
    average = str(average).strip().lower()
    valid = support > 0
    if not bool(valid.any()):
        return 0.0
    if average == "weighted":
        w = support[valid]
        den = float(w.sum().item())
        if den <= 0.0:
            return 0.0
        return float((values[valid] * w).sum().item() / den)
    return float(values[valid].mean().item())


class _RegressionAccumulator:
    def __init__(self) -> None:
        self.sse = 0.0
        self.sae = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.count = 0.0

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

    def mse(self) -> float | None:
        if self.count <= 0.0:
            return None
        return float(self.sse / self.count)

    def mae(self) -> float | None:
        if self.count <= 0.0:
            return None
        return float(self.sae / self.count)

    def rmse(self) -> float | None:
        mse = self.mse()
        if mse is None:
            return None
        return float(torch.sqrt(torch.tensor(mse)).item())

    def r2(self) -> float | None:
        if self.count <= 0.0:
            return None
        mean_y = self.sum_y / self.count
        sst = self.sum_y2 - self.count * mean_y * mean_y
        if sst <= 0.0:
            return 0.0
        return float(1.0 - (self.sse / sst))


@dataclass
class _MetricState:
    scalar: _RunningScalar
    cls: _ClassificationAccumulator
    reg: _RegressionAccumulator


def _new_state() -> _MetricState:
    return _MetricState(
        scalar=_RunningScalar(),
        cls=_ClassificationAccumulator(),
        reg=_RegressionAccumulator(),
    )


class BuiltinEpochMetric(TrainingMetric):
    def __init__(
        self,
        *,
        metric: str,
        output_key: str,
        average: str = "macro",
        positive_label: int = 1,
    ):
        self.metric = str(metric).strip().lower()
        self.output_key = str(output_key).strip()
        self.average = str(average).strip().lower()
        self.positive_label = int(positive_label)
        self._train_state = _new_state()
        self._eval_state = _new_state()
        self._last_value: float | None = None

    def on_epoch_start(self, trainer, epoch: int, state: Any | None = None) -> None:
        self._train_state = _new_state()

    def on_batch_end(
        self,
        trainer,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        self._update_state(self._train_state, stats=stats, batch_size=batch_size)

    def on_epoch_end(self, trainer, epoch: int, state: Any | None = None) -> dict[str, float]:
        value = self._compute_state_value(self._train_state)
        if value is None:
            return {}
        self._last_value = float(value)
        return {self.output_key: self._last_value}

    def on_train_end(self, trainer, state: Any | None = None) -> dict[str, float]:
        if self._last_value is None:
            return {}
        return {self.output_key: float(self._last_value)}

    def on_eval_start(self, trainer, split: str | None, state: Any | None = None) -> None:
        self._eval_state = _new_state()

    def on_eval_batch_end(
        self,
        trainer,
        split: str | None,
        stats: dict[str, Any],
        batch_size: int,
        state: Any | None = None,
    ) -> None:
        self._update_state(self._eval_state, stats=stats, batch_size=batch_size)

    def on_eval_end(
        self,
        trainer,
        split: str | None,
        state: Any | None = None,
    ) -> dict[str, float]:
        value = self._compute_state_value(self._eval_state)
        if value is None:
            return {}
        return {self.output_key: float(value)}

    def _update_state(self, state: _MetricState, *, stats: dict[str, Any], batch_size: int) -> None:
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
        w = float(max(1, int(batch_size)))
        state.scalar.update(fallback, w)

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

    def _compute_state_value(self, state: _MetricState) -> float | None:
        if self.metric == "accuracy":
            return _coalesce(state.cls.accuracy(), state.scalar.value())
        if self.metric == "precision":
            return _coalesce(
                state.cls.precision(average=self.average, positive_label=self.positive_label),
                state.scalar.value(),
            )
        if self.metric == "recall":
            return _coalesce(
                state.cls.recall(average=self.average, positive_label=self.positive_label),
                state.scalar.value(),
            )
        if self.metric == "f1":
            return _coalesce(
                state.cls.f1(average=self.average, positive_label=self.positive_label),
                state.scalar.value(),
            )
        if self.metric == "mse":
            return _coalesce(state.reg.mse(), state.scalar.value())
        if self.metric == "mae":
            return _coalesce(state.reg.mae(), state.scalar.value())
        if self.metric == "rmse":
            return _coalesce(state.reg.rmse(), state.scalar.value())
        if self.metric == "r2":
            return _coalesce(state.reg.r2(), state.scalar.value())
        return None


def _coalesce(primary: float | None, secondary: float | None) -> float | None:
    if primary is not None:
        return primary
    return secondary
