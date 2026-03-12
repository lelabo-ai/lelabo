from __future__ import annotations

import math
import warnings
from abc import ABC
from typing import Any, Sequence

from .display import normalize_display_mode
from .train_types import EpochRecord, FitResult, SplitSummary

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None


class Reporter(ABC):
    def on_run_start(
        self,
        *,
        model_name: str,
        task_name: str,
        rule_name: str,
        device: str,
        epochs: int,
        metric_keys: Sequence[str],
    ) -> None:
        _ = (model_name, task_name, rule_name, device, epochs, metric_keys)

    def on_epoch_end(self, epoch_record: EpochRecord, state: Any | None = None) -> None:
        _ = (epoch_record, state)

    def on_eval_end(self, split_summary: SplitSummary, state: Any | None = None) -> None:
        _ = (split_summary, state)

    def on_run_end(self, fit_result: FitResult, state: Any | None = None) -> None:
        _ = (fit_result, state)


class NullReporter(Reporter):
    pass


class _BaseTextReporter(Reporter):
    def __init__(self) -> None:
        self._console = Console() if (Console is not None and Table is not None) else None
        self._metric_keys: list[str] = []

    @staticmethod
    def _fmt_scalar(key: str, value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "-"
        v = float(value)
        if not math.isfinite(v):
            return "-"
        k = str(key).strip().lower()
        if k == "lr":
            return f"{v:.2e}"
        if "loss" in k:
            return f"{v:.4f}"
        return f"{v:.4f}"

    @staticmethod
    def _human_time(seconds: float) -> str:
        s = max(0, int(round(float(seconds))))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _human_time_short(seconds: float | None) -> str:
        if not isinstance(seconds, (int, float)):
            return "-"
        s = float(seconds)
        if not math.isfinite(s) or s < 0.0:
            return "-"
        return f"{s:.1f}s"

    @staticmethod
    def _format_count(n: int | None) -> str:
        if not isinstance(n, int):
            return "-"
        return f"{n:_}"


class CompactReporter(_BaseTextReporter):
    def on_run_start(
        self,
        *,
        model_name: str,
        task_name: str,
        rule_name: str,
        device: str,
        epochs: int,
        metric_keys: Sequence[str],
    ) -> None:
        self._metric_keys = list(metric_keys)
        metrics = ["loss", *metric_keys] if metric_keys else ["loss", "metric"]
        print(
            f"run | model={model_name} | task={task_name} | rule={rule_name} | "
            f"device={device} | epochs={int(epochs)}"
        )
        print("metrics | " + ", ".join(metrics))

    def on_epoch_end(self, epoch_record: EpochRecord, state: Any | None = None) -> None:
        _ = state
        parts: list[str] = [f"E{int(epoch_record.epoch):03d}"]
        train_fields = [f"loss={self._fmt_scalar('loss', epoch_record.train.loss)}"]
        if epoch_record.train.metric is not None:
            train_fields.append(f"metric={self._fmt_scalar('metric', epoch_record.train.metric)}")
        for key in self._metric_keys:
            if key in epoch_record.train.scalars:
                train_fields.append(f"{key}={self._fmt_scalar(key, epoch_record.train.scalars[key])}")
        if "acc" in epoch_record.train.scalars and "acc" not in self._metric_keys:
            train_fields.append(f"acc={self._fmt_scalar('acc', epoch_record.train.scalars['acc'])}")
        parts.append("train " + " ".join(train_fields))
        if epoch_record.val is not None:
            val_fields = [f"loss={self._fmt_scalar('loss', epoch_record.val.loss)}"]
            if epoch_record.val.metric is not None:
                val_fields.append(f"metric={self._fmt_scalar('metric', epoch_record.val.metric)}")
            for key in self._metric_keys:
                if key in epoch_record.val.scalars:
                    val_fields.append(f"{key}={self._fmt_scalar(key, epoch_record.val.scalars[key])}")
            if "acc" in epoch_record.val.scalars and "acc" not in self._metric_keys:
                val_fields.append(f"acc={self._fmt_scalar('acc', epoch_record.val.scalars['acc'])}")
            parts.append("val " + " ".join(val_fields))
        if epoch_record.lr is not None:
            parts.append(f"lr={self._fmt_scalar('lr', epoch_record.lr)}")
        parts.append(f"t={self._human_time(epoch_record.duration_sec)}")
        print(" | ".join(parts))

    def on_eval_end(self, split_summary: SplitSummary, state: Any | None = None) -> None:
        _ = state
        split_name = str(split_summary.split or "eval")
        fields = [f"loss={self._fmt_scalar('loss', split_summary.loss)}"]
        if split_summary.metric is not None:
            fields.append(f"metric={self._fmt_scalar('metric', split_summary.metric)}")
        print(f"{split_name} | " + " ".join(fields))

    def on_run_end(self, fit_result: FitResult, state: Any | None = None) -> None:
        _ = state
        final_epoch = fit_result.final_epoch
        best_val_str = "best_val=-"
        if fit_result.best.epoch_by_val is not None and fit_result.best.val_metric is not None:
            best_val_str = f"best_val={self._fmt_scalar('val.metric', fit_result.best.val_metric)}@{fit_result.best.epoch_by_val}"
        print(
            f"done | train_loss={self._fmt_scalar('loss', final_epoch.train.loss)} | "
            f"train_metric={self._fmt_scalar('metric', final_epoch.train.metric)} | "
            f"{best_val_str} | total={self._human_time(fit_result.runtime.total_train_time_sec)}"
        )
        if fit_result.runtime.stop_reason:
            print(f"stop | {fit_result.runtime.stop_reason}")


class RichReporter(_BaseTextReporter):
    def __init__(self) -> None:
        super().__init__()
        if self._console is None or Table is None:
            raise RuntimeError("RichReporter requires the 'rich' package.")

    def on_run_start(
        self,
        *,
        model_name: str,
        task_name: str,
        rule_name: str,
        device: str,
        epochs: int,
        metric_keys: Sequence[str],
    ) -> None:
        self._metric_keys = list(metric_keys)
        metrics = ["loss", *metric_keys] if metric_keys else ["loss", "metric"]
        self._console.print(
            f"[bold cyan]run | model={model_name} | task={task_name} | rule={rule_name} | "
            f"device={device} | epochs={int(epochs)}[/]"
        )
        self._console.print(f"[cyan]metrics | {', '.join(metrics)}[/]")

    def on_epoch_end(self, epoch_record: EpochRecord, state: Any | None = None) -> None:
        _ = state
        table = Table(title=f"Epoch {int(epoch_record.epoch)} summary", show_lines=False)
        table.add_column("split", justify="left")
        table.add_column("loss", justify="right")
        table.add_column("metric", justify="right")
        for key in self._metric_keys:
            table.add_column(str(key), justify="right")
        table.add_column("lr", justify="right")
        table.add_column("time", justify="right")
        table.add_column("samples", justify="right")
        train_row = [
            "train",
            self._fmt_scalar("loss", epoch_record.train.loss),
            self._fmt_scalar("metric", epoch_record.train.metric),
        ]
        for key in self._metric_keys:
            train_row.append(self._fmt_scalar(key, epoch_record.train.scalars.get(key)))
        train_row.extend(
            [
                self._fmt_scalar("lr", epoch_record.lr),
                self._human_time_short(epoch_record.train.duration_sec),
                self._format_count(epoch_record.train.num_samples),
            ]
        )
        table.add_row(*train_row)
        if epoch_record.val is not None:
            val_row = [
                "val",
                self._fmt_scalar("loss", epoch_record.val.loss),
                self._fmt_scalar("metric", epoch_record.val.metric),
            ]
            for key in self._metric_keys:
                val_row.append(self._fmt_scalar(key, epoch_record.val.scalars.get(key)))
            val_row.extend(
                [
                    "-",
                    self._human_time_short(epoch_record.val.duration_sec),
                    self._format_count(epoch_record.val.num_samples),
                ]
            )
            table.add_row(*val_row)
        self._console.print(table)
        if epoch_record.monitor is not None:
            best_value = self._fmt_scalar(epoch_record.monitor.name, epoch_record.monitor.best_value)
            line = (
                f"best: {epoch_record.monitor.name}={best_value} "
                f"(epoch {epoch_record.monitor.best_epoch})"
            )
            if epoch_record.monitor.improved_this_epoch:
                line += " ↑"
            self._console.print(line)

    def on_eval_end(self, split_summary: SplitSummary, state: Any | None = None) -> None:
        _ = state
        split_name = str(split_summary.split or "eval")
        self._console.print(
            f"[cyan]{split_name} | loss={self._fmt_scalar('loss', split_summary.loss)} "
            f"metric={self._fmt_scalar('metric', split_summary.metric)}[/]"
        )

    def on_run_end(self, fit_result: FitResult, state: Any | None = None) -> None:
        _ = state
        final_epoch = fit_result.final_epoch
        best_val_str = "best_val=-"
        if fit_result.best.epoch_by_val is not None and fit_result.best.val_metric is not None:
            best_val_str = f"best_val={self._fmt_scalar('val.metric', fit_result.best.val_metric)}@{fit_result.best.epoch_by_val}"
        self._console.print(
            f"[bold green]done | train_loss={self._fmt_scalar('loss', final_epoch.train.loss)} | "
            f"train_metric={self._fmt_scalar('metric', final_epoch.train.metric)} | "
            f"{best_val_str} | total={self._human_time(fit_result.runtime.total_train_time_sec)}[/]"
        )
        if fit_result.runtime.stop_reason:
            self._console.print(f"[yellow]stop | {fit_result.runtime.stop_reason}[/]")


def make_reporter(display_mode: Any) -> tuple[str, Reporter]:
    mode = normalize_display_mode(display_mode if display_mode is not None else "compact", where="display_mode")
    if mode == "none":
        return mode, NullReporter()
    if mode == "compact":
        return mode, CompactReporter()
    if Console is None or Table is None:
        warnings.warn(
            "display='rich' was requested but package 'rich' is not installed. "
            "Falling back to display='compact'. Install it with: pip install rich",
            UserWarning,
            stacklevel=2,
        )
        return "compact", CompactReporter()
    return mode, RichReporter()
