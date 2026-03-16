"""Core supervised training loop and evaluation runtime.

This module hosts :class:`Trainer`, the central runtime object used by
supervised experiments. The trainer is responsible for coordinating the model,
learner, loss, callbacks, metrics, schedulers, structured logging, and console
reporting while producing stable :mod:`lelabo.core.train_types` outputs.
"""

from __future__ import annotations

from dataclasses import replace
import time
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch

from .accumulators import SplitAccumulator
from .batch import infer_batch_size
from .callbacks import Callback, EarlyStopping
from .contracts import (
    validate_callbacks,
    validate_learner,
    validate_logger,
    validate_loss,
    validate_metrics,
    validate_model,
    validate_schedulers,
)
from .logger import RunLogger
from .reporters import make_reporter
from .state import TrainState
from .steps import compute_loss_and_stats, loss_display_name
from .train_types import BestSummary, EpochRecord, FitResult, FitRuntime, MonitorStatus, RestorationStatus, SplitSummary
from ..metrics.base import TrainerMetric
from ..schedulers import SchedulerController


class Trainer:
    """Run supervised training and evaluation with LeLabo's runtime contracts.

    A :class:`Trainer` owns the high-level control flow around one model and one
    learner:

    - validates runtime dependencies up front
    - runs epoch and batch loops for training
    - runs evaluation passes on validation or external loaders
    - updates callbacks, metrics, reporters, and schedulers at the right hooks
    - emits structured run artifacts through :class:`RunLogger`
    - returns stable dataclass summaries such as :class:`FitResult`

    The trainer does not implement the learning algorithm itself. That work is
    delegated to ``learner.train_step(...)`` and the loss object.
    """

    FitResult = FitResult
    SplitSummary = SplitSummary
    EpochRecord = EpochRecord
    MonitorStatus = MonitorStatus
    FitRuntime = FitRuntime
    RestorationStatus = RestorationStatus
    BestSummary = BestSummary

    def __init__(
        self,
        model,
        learner,
        loss,
        device: str = "cpu",
        display_mode: str = "compact",
        callbacks: Optional[list[Callback]] = None,
        logger: Optional[RunLogger] = None,
        schedulers: Optional[list[SchedulerController]] = None,
        metrics: Optional[list[TrainerMetric]] = None,
    ):
        """Initialize a trainer with validated runtime dependencies.

        Args:
            model: Torch module trained and evaluated by the runtime.
            learner: Update-rule object that owns ``train_step(...)`` and any
                optimizer state.
            loss: Loss module or callable consumed by the learner and eval path.
            device: Device string used to place the model and incoming batches.
            display_mode: Reporter mode used for console progress output.
            callbacks: Optional callback hooks invoked during fit and eval.
            logger: Structured logger responsible for artifact persistence.
            schedulers: Optional scheduler controllers stepped on batch or epoch
                boundaries.
            metrics: Optional trainer metrics updated from batch statistics and
                finalized into split and run summaries.
        """
        validate_model(model)
        validate_loss(loss)
        validate_learner(learner)
        validate_callbacks(list(callbacks or []))
        validate_metrics(list(metrics or []))
        validate_schedulers(list(schedulers or []))

        self.model = model.to(device)
        self.learner = learner
        self.loss = loss
        self.loss_name = loss_display_name(loss)
        self.device = device
        self.logger = logger or RunLogger(run_dir=None)
        validate_logger(self.logger)
        self.callbacks = callbacks or []
        self.schedulers = schedulers or []
        self.metrics = metrics or []
        self.stop_training = False
        self.stop_reason: str | None = None
        self.state: TrainState | None = None
        self._in_fit = False

        self.display_mode, self._reporter = make_reporter(display_mode)
        self._display_metric_keys = self._resolve_display_metric_keys()

    @staticmethod
    def _call_callback_hook(callback: Any, hook: str, *args, **kwargs) -> Any:
        """Invoke ``hook`` on ``callback`` when it exists."""
        fn = getattr(callback, hook, None)
        if not callable(fn):
            return None
        return fn(*args, **kwargs)

    def log(self, record: Dict[str, Any]) -> None:
        """Forward one structured log record to the configured logger."""
        self.logger.log(record)

    def request_stop(self, reason: str | None = None) -> None:
        """Request an orderly early stop from callbacks or external control flow.

        The stop is not immediate: the current callback hook or batch finishes,
        then the fit loop exits cleanly with the stop reason recorded in
        :class:`TrainState` and the returned :class:`FitRuntime`.
        """
        self.stop_training = True
        self.stop_reason = None if reason is None else str(reason)
        if self.state is not None:
            self.state.request_stop(self.stop_reason)

    def _resolve_display_metric_keys(self) -> list[str]:
        """Validate and cache metric keys that should be shown by reporters."""
        reserved = {"loss", "metric", "lr"}
        keys: list[str] = []
        seen: set[str] = set()
        for metric in self.metrics:
            display_keys = metric.display_keys()
            if not isinstance(display_keys, (tuple, list)):
                raise TypeError(
                    f"Metric '{type(metric).__name__}' display_keys() must return a tuple/list[str]."
                )
            for key in display_keys:
                token = str(key).strip()
                if not token:
                    raise ValueError(
                        f"Metric '{type(metric).__name__}' returned an empty display key."
                    )
                if token in reserved:
                    raise ValueError(
                        f"Metric display key '{token}' collides with a reserved trainer scalar."
                    )
                if token in seen:
                    raise ValueError(
                        f"Duplicate metric display key '{token}' requested in Trainer.metrics."
                    )
                seen.add(token)
                keys.append(token)
        return keys

    def _current_lr(self) -> float | None:
        """Return the current learning rate from the learner optimizer when available."""
        optimizer = getattr(self.learner, "optimizer", None)
        if not isinstance(optimizer, torch.optim.Optimizer):
            return None
        groups = list(getattr(optimizer, "param_groups", []) or [])
        if not groups:
            return None
        lr = groups[0].get("lr", None)
        return float(lr) if isinstance(lr, (int, float)) else None

    def _step_schedulers(self, *, interval: str, logs: Optional[Dict[str, Any]] = None) -> None:
        """Step all attached scheduler controllers for one interval."""
        if not self.schedulers:
            return
        interval = str(interval).lower()
        for sched in self.schedulers:
            if interval == "batch":
                sched.step_batch(logs=logs)
            elif interval == "epoch":
                sched.step_epoch(logs=logs)

    def _reset_metrics(self, split: str) -> None:
        """Reset all metrics for one split."""
        for metric in self.metrics:
            metric.reset(split, self.state)

    def _update_metrics(self, split: str, *, stats: Mapping[str, Any], batch_size: int) -> None:
        """Update all metrics with one batch worth of statistics."""
        payload = dict(stats)
        for metric in self.metrics:
            metric.update(split, payload, int(batch_size), self.state)

    def _compute_metrics(self, split: str) -> dict[str, float]:
        """Compute current metric values for one split and validate their shapes."""
        out: dict[str, float] = {}
        reserved = {"loss", "metric", "lr"}
        for metric in self.metrics:
            values = metric.compute(split, self.state)
            if not isinstance(values, Mapping):
                raise TypeError(
                    f"Metric '{type(metric).__name__}' compute() must return a mapping[str, float]."
                )
            for key, value in values.items():
                token = str(key).strip()
                if not token:
                    raise ValueError(
                        f"Metric '{type(metric).__name__}' returned an empty key from compute()."
                    )
                if token in reserved:
                    raise ValueError(
                        f"Metric '{type(metric).__name__}' returned reserved key '{token}'."
                    )
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Metric '{type(metric).__name__}' returned non-numeric value for key '{token}'."
                    )
                if token in out:
                    raise ValueError(
                        f"Metric output key collision for '{token}' during split '{split}'."
                    )
                out[token] = float(value)
        return out

    def _finalize_metrics(self) -> dict[str, float]:
        """Finalize metrics at the end of fit and validate their shapes."""
        out: dict[str, float] = {}
        reserved = {"loss", "metric", "lr"}
        for metric in self.metrics:
            values = metric.finalize(self.state)
            if not isinstance(values, Mapping):
                raise TypeError(
                    f"Metric '{type(metric).__name__}' finalize() must return a mapping[str, float]."
                )
            for key, value in values.items():
                token = str(key).strip()
                if not token:
                    raise ValueError(
                        f"Metric '{type(metric).__name__}' returned an empty key from finalize()."
                    )
                if token in reserved:
                    raise ValueError(
                        f"Metric '{type(metric).__name__}' returned reserved finalize key '{token}'."
                    )
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Metric '{type(metric).__name__}' returned non-numeric finalize value for key '{token}'."
                    )
                if token in out:
                    raise ValueError(
                        f"Metric finalize key collision for '{token}'."
                    )
                out[token] = float(value)
        return out

    @staticmethod
    def _train_log_record(epoch_record: EpochRecord) -> dict[str, Any]:
        """Build the canonical epoch-end training record for ``metrics.jsonl``.

        The returned mapping is intentionally flat and JSONL-friendly so that
        downstream analysis tools can consume it without understanding internal
        trainer dataclasses.
        """
        out: dict[str, Any] = {
            "t": "train",
            "event": "epoch_end",
            "split": "train",
            "epoch": int(epoch_record.epoch),
            "loss": float(epoch_record.train.loss),
            "num_samples": int(epoch_record.train.num_samples),
            "num_batches": int(epoch_record.train.num_batches),
            "duration_sec": None if epoch_record.train.duration_sec is None else float(epoch_record.train.duration_sec),
        }
        if epoch_record.train.metric is not None:
            out["metric"] = float(epoch_record.train.metric)
        if epoch_record.lr is not None:
            out["lr"] = float(epoch_record.lr)
        out.update({str(k): float(v) for k, v in epoch_record.train.scalars.items()})
        return out

    @staticmethod
    def _eval_log_record(split_summary: SplitSummary) -> dict[str, Any]:
        """Build the canonical eval-end record for ``metrics.jsonl``.

        Eval records mirror the persisted public run-artifact contract: one flat
        record per split summary, with optional metric and scalar payloads.
        """
        out: dict[str, Any] = {
            "t": "eval",
            "event": "eval_end",
            "split": split_summary.split,
            "loss": float(split_summary.loss),
            "num_samples": int(split_summary.num_samples),
            "num_batches": int(split_summary.num_batches),
            "duration_sec": None if split_summary.duration_sec is None else float(split_summary.duration_sec),
        }
        if split_summary.metric is not None:
            out["metric"] = float(split_summary.metric)
        out.update({str(k): float(v) for k, v in split_summary.scalars.items()})
        return out

    @staticmethod
    def _with_monitor(record: EpochRecord, early_stopping_cb: EarlyStopping | None) -> EpochRecord:
        """Attach monitor status to an epoch record when early stopping is active."""
        if early_stopping_cb is None:
            return record
        status = early_stopping_cb.status()
        return replace(record, monitor=status)

    @staticmethod
    def _monitor_mode_from_name(name: str) -> str:
        """Infer whether a monitor should be minimized or maximized from its name."""
        token = str(name).strip().lower()
        if any(item in token for item in ("acc", "f1", "precision", "recall", "r2", "auc")):
            return "max"
        return "min"

    def _best_record_for_monitor(
        self,
        history: list[EpochRecord],
        *,
        monitor_name: str,
        monitor_mode: str,
    ) -> tuple[EpochRecord, float] | None:
        """Return the best epoch record for one monitor key across the fit history."""
        matches: list[tuple[EpochRecord, float]] = []
        for record in history:
            value = record.to_log_values().get(monitor_name)
            if isinstance(value, (int, float)):
                matches.append((record, float(value)))
        if not matches:
            return None
        if str(monitor_mode).strip().lower() == "max":
            return max(matches, key=lambda item: item[1])
        return min(matches, key=lambda item: item[1])

    def _build_best_summary(self, history: list[EpochRecord], early_stopping_cb: EarlyStopping | None) -> BestSummary:
        """Resolve the best epoch summary using callback, scheduler, or default monitors."""
        if not history:
            raise RuntimeError("Cannot build best summary from an empty history.")

        candidates: list[tuple[str, str, str]] = []
        if early_stopping_cb is not None:
            status = early_stopping_cb.status()
            candidates.append(("earlystopping", str(status.name), str(status.mode)))
        for scheduler in self.schedulers:
            monitor_name = str(getattr(scheduler, "monitor", "")).strip()
            if not monitor_name:
                continue
            raw_mode = getattr(getattr(scheduler, "scheduler", None), "mode", None)
            if str(raw_mode).strip().lower() in {"min", "max"}:
                mode = str(raw_mode).strip().lower()
            else:
                mode = self._monitor_mode_from_name(monitor_name)
            candidates.append(("scheduler", monitor_name, mode))
            break
        if any(record.val is not None for record in history):
            candidates.append(("default", "val.loss", "min"))
        candidates.append(("default", "train.loss", "min"))

        for source, monitor_name, monitor_mode in candidates:
            best_match = self._best_record_for_monitor(
                history,
                monitor_name=monitor_name,
                monitor_mode=monitor_mode,
            )
            if best_match is None:
                continue
            best_record, best_value = best_match
            return BestSummary(
                source=source,
                monitor_name=monitor_name,
                monitor_mode=monitor_mode,
                best_value=float(best_value),
                epoch=int(best_record.epoch),
                train=best_record.train,
                val=best_record.val,
            )

        raise RuntimeError("Trainer could not resolve a valid monitor from fit history.")

    @staticmethod
    def _restoration_status(early_stopping_cb: EarlyStopping | None) -> RestorationStatus:
        """Return the effective restoration status for the fit result."""
        if early_stopping_cb is None:
            return RestorationStatus(
                enabled=False,
                best_epoch=None,
                best_checkpoint_available=False,
                restored_on_train_end=False,
            )
        return early_stopping_cb.restoration_status()

    def fit(self, train_loader, epochs: int = 10, val_loader=None) -> FitResult:
        """Run the full supervised fit loop and return a structured ``FitResult``.

        The fit lifecycle is:

        1. initialize :class:`TrainState` and notify callbacks/logger
        2. iterate over epochs and training batches
        3. collect training summaries and optional validation summaries
        4. step schedulers, callbacks, reporters, and structured logging hooks
        5. resolve the best epoch, runtime metadata, and finalized metrics

        Args:
            train_loader: Iterable of training batches.
            epochs: Maximum number of epochs to run.
            val_loader: Optional validation loader evaluated once per epoch.

        Returns:
            A :class:`FitResult` containing history, best checkpoint metadata,
            runtime status, restoration status, and finalized run metrics.
        """
        self._in_fit = True
        self.stop_training = False
        self.stop_reason = None
        state = TrainState(phase="fit", split="train")
        self.state = state
        self._call_callback_hook(self.logger, "on_fit_start", self, state)
        try:
            train_batches_per_epoch = int(len(train_loader))
            if train_batches_per_epoch <= 0:
                train_batches_per_epoch = None
        except Exception:
            train_batches_per_epoch = None
        total_train_batches = (
            int(epochs) * int(train_batches_per_epoch)
            if train_batches_per_epoch is not None
            else None
        )
        global_batches_completed = 0

        for cb in self.callbacks:
            self._call_callback_hook(cb, "on_train_start", self, state)

        self.learner.on_train_start(self.model, self.loss, self.device, state)

        fit_reporter = self._reporter
        fit_reporter.on_run_start(
            model_name=self.model.__class__.__name__,
            loss_name=self.loss_name,
            rule_name=self.learner.__class__.__name__,
            device=str(self.device),
            epochs=int(epochs),
            metric_keys=self._display_metric_keys,
        )

        early_stopping_cb = next((cb for cb in self.callbacks if isinstance(cb, EarlyStopping)), None)

        history: list[EpochRecord] = []
        train_start = time.perf_counter()

        try:
            for ep in range(1, int(epochs) + 1):
                state.phase = "train"
                state.split = "train"
                state.epoch = int(ep)
                state.batch_idx = 0
                state.last_eval = None
                self.model.train()
                fit_reporter.on_epoch_start(
                    epoch=int(ep),
                    epochs=int(epochs),
                    train_batches=train_batches_per_epoch,
                    global_batches_completed=global_batches_completed,
                    global_batches_total=total_train_batches,
                    state=state,
                )

                train_acc = SplitAccumulator(split="train")
                self._reset_metrics("train")

                for cb in self.callbacks:
                    self._call_callback_hook(cb, "on_epoch_start", self, state)

                ep_start = time.perf_counter()
                for batch_idx, batch in enumerate(train_loader, start=1):
                    state.batch_idx = int(batch_idx)
                    for cb in self.callbacks:
                        self._call_callback_hook(cb, "on_batch_start", self, state)

                    bs = int(max(1, infer_batch_size(batch)))
                    state.train_samples_seen += bs

                    stats = self.learner.train_step(self.model, self.loss, batch, self.device, state)
                    if not isinstance(stats, dict):
                        stats = {}

                    loss = float(stats.get("loss", 0.0))
                    train_acc.update(loss=loss, stats=stats, batch_size=bs)
                    self._update_metrics("train", stats=stats, batch_size=bs)

                    self._step_schedulers(interval="batch", logs=stats)

                    state.current_lr = self._current_lr()
                    global_batches_completed += 1
                    fit_reporter.on_batch_end(
                        epoch=int(ep),
                        epochs=int(epochs),
                        batch_idx=int(batch_idx),
                        train_batches=train_batches_per_epoch,
                        global_batches_completed=global_batches_completed,
                        global_batches_total=total_train_batches,
                        train_summary=train_acc.preview(
                            extra_scalars=self._compute_metrics("train"),
                        ),
                        lr=state.current_lr,
                        state=state,
                    )
                    for cb in self.callbacks:
                        self._call_callback_hook(cb, "on_batch_end", self, state, logs=stats)
                    state.bump_step(1)

                train_duration = float(time.perf_counter() - ep_start)
                train_summary = train_acc.build(
                    duration_sec=train_duration,
                    extra_scalars=self._compute_metrics("train"),
                )
                state.last_train = train_summary

                val_summary: SplitSummary | None = None
                if val_loader is not None:
                    val_summary = self._evaluate(val_loader, split="val", emit_report=False, invoke_callbacks=True)
                    state.last_eval = val_summary

                epoch_logs: Dict[str, Any] = train_summary.to_prefixed_scalars()
                if val_summary is not None:
                    epoch_logs.update(val_summary.to_prefixed_scalars())

                self._step_schedulers(interval="epoch", logs=epoch_logs)
                state.current_lr = self._current_lr()

                epoch_record = EpochRecord(
                    epoch=int(ep),
                    train=train_summary,
                    val=val_summary,
                    lr=state.current_lr,
                    monitor=None,
                    duration_sec=float(train_duration + (val_summary.duration_sec if val_summary is not None and val_summary.duration_sec is not None else 0.0)),
                )

                for cb in self.callbacks:
                    self._call_callback_hook(cb, "on_epoch_end", self, epoch_record, state)

                epoch_record = self._with_monitor(epoch_record, early_stopping_cb)
                state.last_epoch = epoch_record
                self.log(self._train_log_record(epoch_record))
                self._call_callback_hook(self.logger, "on_epoch_end", self, epoch_record, state)
                history.append(epoch_record)
                fit_reporter.on_epoch_end(epoch_record, state)

                if self.stop_training or state.stop_requested:
                    break
        finally:
            self._in_fit = False

        if not history:
            raise RuntimeError("Trainer.fit() completed without producing epoch records.")

        total_time = float(time.perf_counter() - train_start)
        best = self._build_best_summary(history, early_stopping_cb)
        runtime = FitRuntime(
            epochs_completed=len(history),
            total_train_time_sec=total_time,
            stopped_early=bool(self.stop_training or state.stop_requested or len(history) < int(epochs)),
            stop_reason=(state.stop_reason or self.stop_reason),
        )
        fit_result = FitResult(
            history=history,
            final_epoch=history[-1],
            best=best,
            runtime=runtime,
            restoration=self._restoration_status(early_stopping_cb),
            run_metrics=self._finalize_metrics(),
        )

        for cb in self.callbacks:
            self._call_callback_hook(cb, "on_train_end", self, fit_result, state)

        fit_result = FitResult(
            history=history,
            final_epoch=history[-1],
            best=best,
            runtime=runtime,
            restoration=self._restoration_status(early_stopping_cb),
            run_metrics=self._finalize_metrics(),
        )

        self._call_callback_hook(self.logger, "on_fit_end", self, fit_result, state)
        fit_reporter.on_run_end(fit_result, state)
        state.phase = "idle"
        state.split = None
        return fit_result

    @torch.no_grad()
    def _evaluate(
        self,
        loader,
        *,
        split: str | None,
        emit_report: bool,
        invoke_callbacks: bool,
    ) -> SplitSummary:
        """Run one evaluation pass over ``loader`` and return a ``SplitSummary``.

        This internal helper is shared by validation inside :meth:`fit` and by
        the public :meth:`evaluate` entry point. The flags control whether the
        pass should emit human-facing reporter output and callback hooks.
        """
        split_name = str(split or "eval")
        self.model.eval()
        if self.state is None:
            self.state = TrainState()
        state = self.state
        state.phase = "eval"
        state.split = split_name
        state.batch_idx = 0

        acc = SplitAccumulator(split=split_name)
        self._reset_metrics(split_name)
        eval_start = time.perf_counter()

        for batch_idx, batch in enumerate(loader, start=1):
            state.batch_idx = int(batch_idx)
            bs = int(max(1, infer_batch_size(batch)))
            state.eval_samples_seen += bs
            loss, stats = compute_loss_and_stats(self.model, self.loss, batch, self.device)
            if not isinstance(stats, Mapping):
                stats = {}
            stats_dict = dict(stats)
            acc.update(loss=float(loss.item()), stats=stats_dict, batch_size=bs)
            self._update_metrics(split_name, stats=stats_dict, batch_size=bs)

        summary = acc.build(
            duration_sec=float(time.perf_counter() - eval_start),
            extra_scalars=self._compute_metrics(split_name),
        )
        state.last_eval = summary
        eval_record = self._eval_log_record(summary)
        if int(getattr(state, "epoch", 0) or 0) > 0:
            eval_record["epoch"] = int(state.epoch)
        self.log(eval_record)

        if invoke_callbacks:
            for cb in self.callbacks:
                self._call_callback_hook(cb, "on_eval_end", self, summary, state)
        if emit_report and not self._in_fit:
            self._reporter.on_eval_end(summary, state)
        return summary

    @torch.no_grad()
    def evaluate(self, loader, split: Optional[str] = None) -> SplitSummary:
        """Evaluate the current model on ``loader`` outside or inside ``fit``.

        Args:
            loader: Iterable of evaluation batches.
            split: Optional split name stored in logs and summaries. Defaults to
                ``"eval"`` when omitted.

        Returns:
            A :class:`SplitSummary` with aggregate loss, optional metric, scalar
            payloads, and runtime counters for the requested split.
        """
        summary = self._evaluate(loader, split=split, emit_report=True, invoke_callbacks=True)
        if self.state is not None:
            self.state.phase = "idle" if not self._in_fit else "fit"
            if not self._in_fit:
                self.state.split = None
        return summary
