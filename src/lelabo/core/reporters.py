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

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class Reporter(ABC):
    def on_run_start(
        self,
        *,
        model_name: str,
        loss_name: str,
        rule_name: str,
        device: str,
        epochs: int,
        metric_keys: Sequence[str],
    ) -> None:
        _ = (model_name, loss_name, rule_name, device, epochs, metric_keys)

    def on_epoch_end(self, epoch_record: EpochRecord, state: Any | None = None) -> None:
        _ = (epoch_record, state)

    def on_epoch_start(
        self,
        *,
        epoch: int,
        epochs: int,
        train_batches: int | None,
        global_batches_completed: int,
        global_batches_total: int | None,
        state: Any | None = None,
    ) -> None:
        _ = (epoch, epochs, train_batches, global_batches_completed, global_batches_total, state)

    def on_batch_end(
        self,
        *,
        epoch: int,
        epochs: int,
        batch_idx: int,
        train_batches: int | None,
        global_batches_completed: int,
        global_batches_total: int | None,
        train_summary: SplitSummary,
        lr: float | None,
        state: Any | None = None,
    ) -> None:
        _ = (
            epoch,
            epochs,
            batch_idx,
            train_batches,
            global_batches_completed,
            global_batches_total,
            train_summary,
            lr,
            state,
        )

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
        self._progress_active = False
        self._global_bar: Any | None = None
        self._epoch_bar: Any | None = None
        self._epochs_total: int = 0

    def _is_progress_enabled(self) -> bool:
        return tqdm is not None

    def _emit_text(self, line: str) -> None:
        text = str(line)
        if self._progress_active and self._is_progress_enabled():
            tqdm.write(text)
            return
        print(text)

    def _emit_rich(self, renderable: Any) -> None:
        if self._console is None:
            self._emit_text(str(renderable))
            return
        if self._progress_active and self._is_progress_enabled():
            with tqdm.external_write_mode():
                self._console.print(renderable)
            return
        self._console.print(renderable)

    def _ensure_global_progress(
        self,
        *,
        global_batches_total: int | None,
        global_batches_completed: int,
    ) -> None:
        if not self._is_progress_enabled() or self._global_bar is not None:
            return
        self._global_bar = tqdm(
            total=global_batches_total,
            initial=int(max(0, global_batches_completed)),
            desc="Global",
            leave=True,
            position=0,
            dynamic_ncols=True,
        )
        self._progress_active = True

    def _open_epoch_progress(
        self,
        *,
        epoch: int,
        epochs: int,
        train_batches: int | None,
    ) -> None:
        if not self._is_progress_enabled():
            return
        if self._epoch_bar is not None:
            self._epoch_bar.close()
        self._epoch_bar = tqdm(
            total=train_batches,
            desc=f"E{int(epoch):03d}/{int(epochs):03d}",
            leave=False,
            position=1,
            dynamic_ncols=True,
        )
        self._progress_active = True

    def _close_epoch_progress(self) -> None:
        if self._epoch_bar is not None:
            self._epoch_bar.close()
            self._epoch_bar = None

    def _close_all_progress(self) -> None:
        self._close_epoch_progress()
        if self._global_bar is not None:
            self._global_bar.close()
            self._global_bar = None
        self._progress_active = False

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

    def _progress_metric_value(self, train_summary: SplitSummary) -> float | None:
        if isinstance(train_summary.metric, (int, float)):
            return float(train_summary.metric)
        if isinstance(train_summary.scalars.get("acc"), (int, float)):
            return float(train_summary.scalars["acc"])
        return None

    def _batch_postfix(
        self,
        *,
        train_summary: SplitSummary,
        lr: float | None,
    ) -> dict[str, str]:
        postfix: dict[str, str] = {
            "loss": self._fmt_scalar("loss", train_summary.loss),
        }
        metric_value = self._progress_metric_value(train_summary)
        if metric_value is not None:
            postfix["metric"] = self._fmt_scalar("metric", metric_value)
        for key in self._metric_keys:
            if key in train_summary.scalars:
                postfix[str(key)] = self._fmt_scalar(key, train_summary.scalars[key])
        if isinstance(lr, (int, float)):
            postfix["lr"] = self._fmt_scalar("lr", lr)
        return postfix


class CompactReporter(_BaseTextReporter):
    def on_run_start(
        self,
        *,
        model_name: str,
        loss_name: str,
        rule_name: str,
        device: str,
        epochs: int,
        metric_keys: Sequence[str],
    ) -> None:
        self._metric_keys = list(metric_keys)
        self._epochs_total = int(epochs)
        metrics = ["loss", *metric_keys] if metric_keys else ["loss", "metric"]
        self._emit_text(
            f"run | model={model_name} | loss={loss_name} | rule={rule_name} | "
            f"device={device} | epochs={int(epochs)}"
        )
        self._emit_text("metrics | " + ", ".join(metrics))

    def on_epoch_start(
        self,
        *,
        epoch: int,
        epochs: int,
        train_batches: int | None,
        global_batches_completed: int,
        global_batches_total: int | None,
        state: Any | None = None,
    ) -> None:
        _ = state
        self._epochs_total = int(epochs)
        self._ensure_global_progress(
            global_batches_total=global_batches_total,
            global_batches_completed=global_batches_completed,
        )
        self._open_epoch_progress(
            epoch=epoch,
            epochs=epochs,
            train_batches=train_batches,
        )

    def on_batch_end(
        self,
        *,
        epoch: int,
        epochs: int,
        batch_idx: int,
        train_batches: int | None,
        global_batches_completed: int,
        global_batches_total: int | None,
        train_summary: SplitSummary,
        lr: float | None,
        state: Any | None = None,
    ) -> None:
        _ = (batch_idx, train_batches, global_batches_total, state)
        if self._epoch_bar is not None:
            self._epoch_bar.update(1)
            self._epoch_bar.set_postfix(self._batch_postfix(train_summary=train_summary, lr=lr))
        if self._global_bar is not None:
            self._global_bar.update(1)
            self._global_bar.set_postfix(epoch=f"{int(epoch)}/{int(epochs)}")

    def on_epoch_end(self, epoch_record: EpochRecord, state: Any | None = None) -> None:
        _ = state
        self._close_epoch_progress()
        epoch_token = f"E{int(epoch_record.epoch):03d}"
        if self._epochs_total > 0:
            epoch_token = f"{epoch_token}/{int(self._epochs_total):03d}"
        parts: list[str] = [epoch_token]
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
        self._emit_text(" | ".join(parts))

    def on_eval_end(self, split_summary: SplitSummary, state: Any | None = None) -> None:
        _ = state
        split_name = str(split_summary.split or "eval")
        fields = [f"loss={self._fmt_scalar('loss', split_summary.loss)}"]
        if split_summary.metric is not None:
            fields.append(f"metric={self._fmt_scalar('metric', split_summary.metric)}")
        self._emit_text(f"{split_name} | " + " ".join(fields))

    def on_run_end(self, fit_result: FitResult, state: Any | None = None) -> None:
        _ = state
        self._close_all_progress()
        final_epoch = fit_result.final_epoch
        best_str = (
            f"best_{fit_result.best.monitor_name}="
            f"{self._fmt_scalar(fit_result.best.monitor_name, fit_result.best.best_value)}"
            f"@{fit_result.best.epoch}"
        )
        self._emit_text(
            f"done | train_loss={self._fmt_scalar('loss', final_epoch.train.loss)} | "
            f"train_metric={self._fmt_scalar('metric', final_epoch.train.metric)} | "
            f"{best_str} | total={self._human_time(fit_result.runtime.total_train_time_sec)}"
        )
        if fit_result.runtime.stop_reason:
            self._emit_text(f"stop | {fit_result.runtime.stop_reason}")
        if fit_result.restoration.restored_on_train_end:
            self._emit_text(f"restore | best_epoch={fit_result.restoration.best_epoch}")


class RichReporter(_BaseTextReporter):
    def __init__(self) -> None:
        super().__init__()
        if self._console is None or Table is None:
            raise RuntimeError("RichReporter requires the 'rich' package.")

    def on_run_start(
        self,
        *,
        model_name: str,
        loss_name: str,
        rule_name: str,
        device: str,
        epochs: int,
        metric_keys: Sequence[str],
    ) -> None:
        self._metric_keys = list(metric_keys)
        self._epochs_total = int(epochs)
        metrics = ["loss", *metric_keys] if metric_keys else ["loss", "metric"]
        self._emit_rich(
            f"[bold cyan]run | model={model_name} | loss={loss_name} | rule={rule_name} | "
            f"device={device} | epochs={int(epochs)}[/]"
        )
        self._emit_rich(f"[cyan]metrics | {', '.join(metrics)}[/]")

    def on_epoch_start(
        self,
        *,
        epoch: int,
        epochs: int,
        train_batches: int | None,
        global_batches_completed: int,
        global_batches_total: int | None,
        state: Any | None = None,
    ) -> None:
        _ = state
        self._epochs_total = int(epochs)
        self._ensure_global_progress(
            global_batches_total=global_batches_total,
            global_batches_completed=global_batches_completed,
        )
        self._open_epoch_progress(
            epoch=epoch,
            epochs=epochs,
            train_batches=train_batches,
        )

    def on_batch_end(
        self,
        *,
        epoch: int,
        epochs: int,
        batch_idx: int,
        train_batches: int | None,
        global_batches_completed: int,
        global_batches_total: int | None,
        train_summary: SplitSummary,
        lr: float | None,
        state: Any | None = None,
    ) -> None:
        _ = (batch_idx, train_batches, global_batches_total, state)
        if self._epoch_bar is not None:
            self._epoch_bar.update(1)
            self._epoch_bar.set_postfix(self._batch_postfix(train_summary=train_summary, lr=lr))
        if self._global_bar is not None:
            self._global_bar.update(1)
            self._global_bar.set_postfix(epoch=f"{int(epoch)}/{int(epochs)}")

    def on_epoch_end(self, epoch_record: EpochRecord, state: Any | None = None) -> None:
        _ = state
        self._close_epoch_progress()
        title = f"Epoch {int(epoch_record.epoch)} summary"
        if self._epochs_total > 0:
            title = f"Epoch {int(epoch_record.epoch)}/{int(self._epochs_total)} summary"
        table = Table(title=title, show_lines=False)
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
        self._emit_rich(table)
        if epoch_record.monitor is not None:
            best_value = self._fmt_scalar(epoch_record.monitor.name, epoch_record.monitor.best_value)
            line = (
                f"best: {epoch_record.monitor.name}={best_value} "
                f"(epoch {epoch_record.monitor.best_epoch})"
            )
            if epoch_record.monitor.improved_this_epoch:
                line += " ↑"
            self._emit_rich(line)

    def on_eval_end(self, split_summary: SplitSummary, state: Any | None = None) -> None:
        _ = state
        split_name = str(split_summary.split or "eval")
        self._emit_rich(
            f"[cyan]{split_name} | loss={self._fmt_scalar('loss', split_summary.loss)} "
            f"metric={self._fmt_scalar('metric', split_summary.metric)}[/]"
        )

    def on_run_end(self, fit_result: FitResult, state: Any | None = None) -> None:
        _ = state
        self._close_all_progress()
        final_epoch = fit_result.final_epoch
        best_str = (
            f"best_{fit_result.best.monitor_name}="
            f"{self._fmt_scalar(fit_result.best.monitor_name, fit_result.best.best_value)}"
            f"@{fit_result.best.epoch}"
        )
        self._emit_rich(
            f"[bold green]done | train_loss={self._fmt_scalar('loss', final_epoch.train.loss)} | "
            f"train_metric={self._fmt_scalar('metric', final_epoch.train.metric)} | "
            f"{best_str} | total={self._human_time(fit_result.runtime.total_train_time_sec)}[/]"
        )
        if fit_result.runtime.stop_reason:
            self._emit_rich(f"[yellow]stop | {fit_result.runtime.stop_reason}[/]")
        if fit_result.restoration.restored_on_train_end:
            self._emit_rich(
                f"[yellow]restore | best_epoch={fit_result.restoration.best_epoch}[/]"
            )


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
