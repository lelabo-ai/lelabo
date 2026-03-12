# lab/core/trainer.py
from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch

from .batch import infer_batch_size
from .callbacks import Callback, EarlyStopping
from .display import normalize_display_mode
from .logger import RunLogger
from .steps import compute_loss_and_stats
from .state import TrainState
from ..metrics.payload import is_metric_payload_key
from ..schedulers import SchedulerController

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None


class Trainer:
    def __init__(
        self,
        model,
        task,
        learner,
        device: str = "cpu",
        input_noise_training: float = 0.0,
        display_mode: str = "compact",
        callbacks: Optional[list[Callback]] = None,
        logger: Optional[RunLogger] = None,
        schedulers: Optional[list[Any]] = None,
        scheduler_interval: str = "epoch",
        scheduler_monitor: str = "val.loss",
        metric_probes: Optional[list[Any]] = None,
    ):
        self.model = model.to(device)
        self.task = task
        self.input_noise_training = float(input_noise_training)

        self.learner = learner

        self.device = device
        self.display_mode = self._normalize_display_mode(display_mode)
        if self.display_mode == "rich" and (Console is None or Table is None):
            self.display_mode = "compact"
            warnings.warn(
                "display='rich' was requested but package 'rich' is not installed. "
                "Falling back to display='compact'. Install it with: pip install rich",
                UserWarning,
                stacklevel=2,
            )
        self._console = Console() if (self.display_mode == "rich" and Console is not None) else None
        self.logger = logger or RunLogger(run_dir=None)
        self.callbacks = callbacks or []
        self.schedulers = schedulers or []
        self.scheduler_interval = str(scheduler_interval).lower()
        self.scheduler_monitor = str(scheduler_monitor)
        self.metric_probes = metric_probes or []
        self.stop_training = False
        self.stop_reason = None
        self.state: TrainState | None = None
        self._in_fit = False
        self._progress_active = False
        self._display_metric_keys = self._resolve_display_metric_keys()

    @staticmethod
    def _normalize_display_mode(mode: Any) -> str:
        token = mode if mode is not None else "compact"
        return normalize_display_mode(token, where="display_mode")

    def _is_silent(self) -> bool:
        return self.display_mode == "none"

    def _is_rich(self) -> bool:
        return self.display_mode == "rich" and self._console is not None and Table is not None

    def _emit_text(self, line: str) -> None:
        text = str(line)
        if self._progress_active and (tqdm is not None):
            tqdm.write(text)
            return
        print(text)

    def _emit_rich(self, renderable: Any) -> None:
        if not self._is_rich():
            self._emit_text(str(renderable))
            return
        if self._progress_active and (tqdm is not None):
            with tqdm.external_write_mode():
                self._console.print(renderable)
            return
        self._console.print(renderable)

    @staticmethod
    def _call_probe_hook(probe: Any, hook: str, *args) -> Any:
        fn = getattr(probe, hook, None)
        if not callable(fn):
            return None
        return fn(*args)

    @staticmethod
    def _call_callback_hook(callback: Any, hook: str, *args, **kwargs) -> Any:
        fn = getattr(callback, hook, None)
        if not callable(fn):
            return None
        return fn(*args, **kwargs)

    def log(self, record: Dict[str, Any]) -> None:
        self.logger.log(record)

    def _resolve_display_metric_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for probe in self.metric_probes:
            key = getattr(probe, "output_key", None)
            if not isinstance(key, str):
                continue
            token = str(key).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            keys.append(token)
        return keys

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
    def _loader_num_samples(loader: Any) -> int | None:
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return None
        try:
            n = int(len(dataset))
        except Exception:
            return None
        return n if n >= 0 else None

    @staticmethod
    def _format_count(n: int | None) -> str:
        if not isinstance(n, int):
            return "-"
        return f"{n:_}"

    def _current_lr(self) -> float | None:
        optimizer = getattr(self.learner, "optimizer", None)
        if not isinstance(optimizer, torch.optim.Optimizer):
            return None
        groups = list(getattr(optimizer, "param_groups", []) or [])
        if not groups:
            return None
        lr = groups[0].get("lr", None)
        return float(lr) if isinstance(lr, (int, float)) else None

    def _print_run_header(self, *, epochs: int) -> None:
        if self._is_silent():
            return
        model_name = self.model.__class__.__name__
        rule_name = self.learner.__class__.__name__
        task_name = self.task.__class__.__name__
        metrics = ["loss", *self._display_metric_keys] if self._display_metric_keys else ["loss", "metric"]
        line_1 = (
            f"run | model={model_name} | task={task_name} | rule={rule_name} | "
            f"device={self.device} | epochs={int(epochs)}"
        )
        line_2 = "metrics | " + ", ".join(metrics)
        if self._is_rich():
            self._emit_rich(f"[bold cyan]{line_1}[/]")
            self._emit_rich(f"[cyan]{line_2}[/]")
            return
        self._emit_text(line_1)
        self._emit_text(line_2)

    def _print_epoch_compact(
        self,
        *,
        epoch: int,
        epochs: int,
        train_loss: float,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float] | None,
        epoch_time_sec: float,
        lr: float | None,
    ) -> None:
        if self._is_silent():
            return
        parts: list[str] = [f"E{int(epoch):03d}/{int(epochs):03d}"]

        train_fields = [f"loss={self._fmt_scalar('loss', train_loss)}"]
        keys = self._display_metric_keys if self._display_metric_keys else ["metric"]
        for key in keys:
            if key not in train_metrics:
                continue
            train_fields.append(f"{key}={self._fmt_scalar(key, train_metrics[key])}")
        parts.append("train " + " ".join(train_fields))

        if isinstance(val_metrics, Mapping) and val_metrics:
            val_fields = [f"loss={self._fmt_scalar('loss', val_metrics.get('loss'))}"]
            for key in keys:
                if key not in val_metrics:
                    continue
                val_fields.append(f"{key}={self._fmt_scalar(key, val_metrics[key])}")
            parts.append("val " + " ".join(val_fields))

        if isinstance(lr, (int, float)):
            parts.append(f"lr={self._fmt_scalar('lr', lr)}")
        parts.append(f"t={self._human_time(epoch_time_sec)}")
        line = " | ".join(parts)
        if self._is_rich():
            line = (
                line.replace("train ", "[green]train[/] ")
                .replace("val ", "[magenta]val[/] ")
                .replace("lr=", "[yellow]lr[/]=")
            )
            self._emit_rich(line)
            return
        self._emit_text(line)

    def _print_epoch_rich(
        self,
        *,
        epoch: int,
        train_loss: float,
        train_metrics: Mapping[str, float],
        lr: float | None,
        train_time_sec: float,
        train_samples: int,
        val_metrics: Mapping[str, float] | None,
        val_time_sec: float | None,
        val_samples: int | None,
        best_monitor_name: str,
        best_monitor_value: float | None,
        best_monitor_epoch: int | None,
        best_improved_this_epoch: bool,
        early_stopping_line: str | None,
        checkpoint_line: str | None,
    ) -> None:
        if not self._is_rich():
            return

        keys = self._display_metric_keys if self._display_metric_keys else ["metric"]
        table = Table(title=f"Epoch {int(epoch)} summary", show_lines=False)
        table.add_column("split", justify="left")
        table.add_column("loss", justify="right")
        for key in keys:
            table.add_column(str(key), justify="right")
        table.add_column("lr", justify="right")
        table.add_column("time", justify="right")
        table.add_column("samples", justify="right")

        train_row = ["train", self._fmt_scalar("loss", train_loss)]
        for key in keys:
            train_row.append(self._fmt_scalar(key, train_metrics.get(key)))
        train_row.append(self._fmt_scalar("lr", lr))
        train_row.append(self._human_time_short(train_time_sec))
        train_row.append(self._format_count(int(train_samples)))
        table.add_row(*train_row)

        if isinstance(val_metrics, Mapping) and val_metrics:
            val_row = ["val", self._fmt_scalar("loss", val_metrics.get("loss"))]
            for key in keys:
                val_row.append(self._fmt_scalar(key, val_metrics.get(key)))
            val_row.append("-")
            val_row.append(self._human_time_short(val_time_sec))
            val_row.append(self._format_count(val_samples))
            table.add_row(*val_row)

        self._emit_rich(table)

        if isinstance(best_monitor_value, (int, float)) and isinstance(best_monitor_epoch, int):
            arrow = " ↑" if best_improved_this_epoch else ""
            self._emit_rich(
                f"best: {best_monitor_name}={self._fmt_scalar(best_monitor_name, best_monitor_value)} "
                f"(epoch {best_monitor_epoch}){arrow}"
            )
        else:
            self._emit_rich(f"best: {best_monitor_name}=-")

        if early_stopping_line:
            self._emit_rich(early_stopping_line)
        if checkpoint_line:
            self._emit_rich(checkpoint_line)

    def _maybe_sync_cuda(self) -> None:
        if isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _step_schedulers(self, *, interval: str, logs: Optional[Dict[str, Any]] = None) -> None:
        if not self.schedulers:
            return
        interval = str(interval).lower()
        for sched in self.schedulers:
            if not isinstance(sched, SchedulerController):
                raise TypeError(
                    "Trainer.schedulers expects SchedulerController instances. "
                    "Build schedulers via lelabo.schedulers.make_scheduler(...)."
                )
            if interval == "batch":
                sched.step_batch(logs=logs)
            elif interval == "epoch":
                sched.step_epoch(logs=logs)

    def fit(self, train_loader, epochs: int = 10, show_progress: bool = True, val_loader=None) -> Dict[str, Any]:
        self._in_fit = True
        best_val_metric = float("-inf")
        best_val_loss = float("inf")
        best_epoch_by_val = None

        best_train_metric = float("-inf")
        best_train_loss = float("inf")

        final_train_loss = None
        final_train_metric = None
        final_val_metric = None
        final_val_loss = None

        state = TrainState()
        self.state = state

        for cb in self.callbacks:
            self._call_callback_hook(cb, "on_train_start", self, state)

        self.learner.on_train_start(self.model, self.task, self.device, state)
        for probe in self.metric_probes:
            self._call_probe_hook(probe, "on_train_start", self, state)
        if not self._is_silent():
            self._print_run_header(epochs=epochs)

        early_stopping_cb = None
        for cb in self.callbacks:
            if isinstance(cb, EarlyStopping):
                early_stopping_cb = cb
                break

        monitor_name = "val.metric" if val_loader is not None else "train.metric"
        monitor_mode = "max"
        if early_stopping_cb is not None:
            cfg = getattr(early_stopping_cb, "cfg", None)
            if cfg is not None:
                monitor_name = str(getattr(cfg, "monitor", monitor_name))
                monitor_mode = str(getattr(cfg, "mode", "max")).strip().lower()
        if monitor_mode not in {"max", "min"}:
            monitor_mode = "min" if "loss" in monitor_name.lower() else "max"
        monitor_best = float("-inf") if monitor_mode == "max" else float("inf")
        monitor_best_epoch: int | None = None

        val_samples_fixed = self._loader_num_samples(val_loader) if val_loader is not None else None

        last_epoch_custom_metrics: dict[str, float] = {}

        self._maybe_sync_cuda()
        train_start = time.perf_counter()

        last_epoch_ran = 0
        use_bar = bool(show_progress and (tqdm is not None) and (not self._is_silent()))
        steps_per_epoch: int | None = None
        if use_bar:
            try:
                steps_per_epoch = int(len(train_loader))  # type: ignore[arg-type]
                if steps_per_epoch <= 0:
                    steps_per_epoch = None
            except Exception:
                steps_per_epoch = None
        global_bar = None
        if use_bar:
            total_steps = (int(epochs) * int(steps_per_epoch)) if steps_per_epoch is not None else None
            global_bar = tqdm(
                total=total_steps,
                desc="Global",
                leave=True,
                position=0,
                dynamic_ncols=True,
            )
        self._progress_active = bool(use_bar)

        try:
            for ep in range(1, epochs + 1):
                last_epoch_ran = ep
                state.epoch = int(ep)
                state.batch_idx = 0
                self.model.train()

                self._maybe_sync_cuda()
                ep_start = time.perf_counter()

                ep_loss_sum = 0.0
                ep_metric_sum = 0.0
                ep_samples = 0
                ep_batches = 0

                for cb in self.callbacks:
                    self._call_callback_hook(cb, "on_epoch_start", self, state)
                for probe in self.metric_probes:
                    self._call_probe_hook(probe, "on_epoch_start", self, ep, state)

                iterator = train_loader
                epoch_bar = None
                if use_bar:
                    epoch_bar = tqdm(
                        total=steps_per_epoch,
                        desc=f"Epoch {ep}/{epochs}",
                        leave=False,
                        position=1,
                        dynamic_ncols=True,
                    )

                try:
                    for batch_idx, batch in enumerate(iterator, start=1):
                        state.batch_idx = int(batch_idx)
                        for cb in self.callbacks:
                            self._call_callback_hook(cb, "on_batch_start", self, state)
                        bs = infer_batch_size(batch)
                        ep_samples += bs

                        if self.input_noise_training > 0.0 and isinstance(batch, (tuple, list)) and len(batch) == 2:
                            x, y = batch
                            if torch.is_tensor(x) and x.is_floating_point():
                                x = x.to(self.device)
                                x = x + torch.randn_like(x) * self.input_noise_training
                                batch = (x, y)

                        stats = self.learner.train_step(self.model, self.task, batch, self.device, state)
                        if not isinstance(stats, dict):
                            stats = {}
                        for probe in self.metric_probes:
                            self._call_probe_hook(probe, "on_batch_end", self, stats, int(max(1, bs)), state)

                        loss = float(stats.get("loss", 0.0))
                        metric = float(stats.get("acc", stats.get("agg", stats.get("metric", 0.0))))

                        w = float(bs if bs > 0 else 1)
                        ep_loss_sum += loss * w
                        ep_metric_sum += metric * w
                        ep_batches += 1

                        denom = max(1.0, float(ep_samples))
                        if epoch_bar is not None:
                            epoch_bar.update(1)
                            epoch_bar.set_postfix(
                                loss=f"{(ep_loss_sum / denom):.4f}",
                                metric=f"{(ep_metric_sum / denom):.4f}",
                            )
                        if global_bar is not None:
                            global_bar.update(1)
                            global_bar.set_postfix(epoch=f"{ep}/{epochs}")

                        self._step_schedulers(interval="batch", logs=stats)

                        for cb in self.callbacks:
                            self._call_callback_hook(cb, "on_batch_end", self, state, logs=stats)

                        state.bump_step(1)
                finally:
                    if epoch_bar is not None:
                        epoch_bar.close()

                self._maybe_sync_cuda()
                ep_time = time.perf_counter() - ep_start

                denom = max(1.0, float(ep_samples))
                mean_loss = ep_loss_sum / denom
                mean_metric = ep_metric_sum / denom

                best_train_metric = max(best_train_metric, mean_metric)
                best_train_loss = min(best_train_loss, mean_loss)
                final_train_loss = float(mean_loss)
                final_train_metric = float(mean_metric)

                epoch_custom_metrics: dict[str, float] = {}
                for probe in self.metric_probes:
                    out = self._call_probe_hook(probe, "on_epoch_end", self, ep, state)
                    if not isinstance(out, dict):
                        continue
                    for key, value in out.items():
                        if isinstance(value, (int, float)):
                            epoch_custom_metrics[str(key)] = float(value)
                last_epoch_custom_metrics = dict(epoch_custom_metrics)

                train_record: Dict[str, Any] = {"t": "train", "epoch": ep, "loss": float(mean_loss), "metric": float(mean_metric)}
                train_record.update(epoch_custom_metrics)
                self.log(train_record)

                logs: Dict[str, float] = {"train.loss": float(mean_loss), "train.metric": float(mean_metric)}
                for key, value in epoch_custom_metrics.items():
                    logs[f"train.{key}"] = float(value)

                val_time_sec: float | None = None
                if val_loader is not None:
                    self._maybe_sync_cuda()
                    val_start = time.perf_counter()
                    val_res = self.evaluate(val_loader, split="val")
                    self._maybe_sync_cuda()
                    val_time_sec = float(time.perf_counter() - val_start)
                    vloss = float(val_res.get("loss", 0.0))
                    vmetric = float(val_res.get("acc", val_res.get("agg", val_res.get("metric", 0.0))))

                    for key, value in val_res.items():
                        if isinstance(value, (int, float)):
                            logs[f"val.{key}"] = float(value)
                    logs.setdefault("val.loss", vloss)
                    logs.setdefault("val.metric", vmetric)

                    for cb in self.callbacks:
                        self._call_callback_hook(cb, "on_eval_end", self, val_res, state)

                    if vmetric > best_val_metric:
                        best_val_metric = vmetric
                        best_val_loss = vloss
                        best_epoch_by_val = ep

                    final_val_metric = vmetric
                    final_val_loss = vloss
                else:
                    val_res = None

                for cb in self.callbacks:
                    self._call_callback_hook(cb, "on_epoch_end", self, ep, logs, state)

                self._step_schedulers(interval="epoch", logs=logs)

                monitor_improved = False
                monitor_value = logs.get(monitor_name)
                if isinstance(monitor_value, (int, float)):
                    v = float(monitor_value)
                    if monitor_best_epoch is None:
                        monitor_best = v
                        monitor_best_epoch = int(ep)
                        monitor_improved = True
                    elif (monitor_mode == "max" and v > monitor_best) or (monitor_mode == "min" and v < monitor_best):
                        monitor_best = v
                        monitor_best_epoch = int(ep)
                        monitor_improved = True

                early_line = None
                checkpoint_line = None
                if early_stopping_cb is not None:
                    cfg = getattr(early_stopping_cb, "cfg", None)
                    bad_epochs = int(getattr(early_stopping_cb, "bad_epochs", 0))
                    patience = int(getattr(cfg, "patience", 0)) if cfg is not None else 0
                    monitor_label = str(getattr(cfg, "monitor", monitor_name)) if cfg is not None else str(monitor_name)
                    mode_label = str(getattr(cfg, "mode", monitor_mode)) if cfg is not None else str(monitor_mode)
                    early_line = (
                        f"early_stopping: monitor={monitor_label} mode={mode_label} "
                        f"patience={patience} wait={bad_epochs}/{patience}"
                    )
                    restore_best = bool(getattr(cfg, "restore_best", False)) if cfg is not None else False
                    has_best_state = getattr(early_stopping_cb, "best_state", None) is not None
                    if restore_best and has_best_state:
                        checkpoint_line = "checkpoint: saved=best"
                    elif restore_best:
                        checkpoint_line = "checkpoint: saved=none"
                    else:
                        checkpoint_line = "checkpoint: disabled"

                if not self._is_silent():
                    train_display = dict(epoch_custom_metrics)
                    if "metric" not in train_display:
                        train_display["metric"] = float(mean_metric)
                    val_display: dict[str, float] | None = None
                    if isinstance(val_res, Mapping):
                        val_display = {}
                        for key, value in val_res.items():
                            if isinstance(value, (int, float)):
                                val_display[str(key)] = float(value)
                    if self._is_rich():
                        self._print_epoch_rich(
                            epoch=ep,
                            train_loss=float(mean_loss),
                            train_metrics=train_display,
                            lr=self._current_lr(),
                            train_time_sec=float(ep_time),
                            train_samples=int(ep_samples),
                            val_metrics=val_display,
                            val_time_sec=val_time_sec,
                            val_samples=val_samples_fixed,
                            best_monitor_name=str(monitor_name),
                            best_monitor_value=(None if monitor_best_epoch is None else float(monitor_best)),
                            best_monitor_epoch=monitor_best_epoch,
                            best_improved_this_epoch=bool(monitor_improved),
                            early_stopping_line=early_line,
                            checkpoint_line=checkpoint_line,
                        )
                    else:
                        self._print_epoch_compact(
                            epoch=ep,
                            epochs=epochs,
                            train_loss=float(mean_loss),
                            train_metrics=train_display,
                            val_metrics=val_display,
                            epoch_time_sec=float(ep_time),
                            lr=self._current_lr(),
                        )
                if self.stop_training:
                    break
        finally:
            self._progress_active = False
            if global_bar is not None:
                global_bar.close()

        self._maybe_sync_cuda()
        total_time = time.perf_counter() - train_start

        final_custom_metrics: dict[str, float] = {}
        for probe in self.metric_probes:
            out = self._call_probe_hook(probe, "on_train_end", self, state)
            if not isinstance(out, dict):
                continue
            for key, value in out.items():
                if isinstance(value, (int, float)):
                    final_custom_metrics[str(key)] = float(value)
        if not final_custom_metrics and last_epoch_custom_metrics:
            final_custom_metrics = dict(last_epoch_custom_metrics)

        for cb in self.callbacks:
            self._call_callback_hook(cb, "on_train_end", self, {"last_epoch": float(last_epoch_ran)}, state)
        self._in_fit = False

        restored_best_model = False
        restored_best_epoch: int | None = None
        if early_stopping_cb is not None:
            cfg = getattr(early_stopping_cb, "cfg", None)
            has_best_state = getattr(early_stopping_cb, "best_state", None) is not None
            if bool(getattr(cfg, "restore_best", False)) and has_best_state:
                restored_best_model = True
                best_epoch = getattr(early_stopping_cb, "best_epoch", None)
                if isinstance(best_epoch, int) and best_epoch > 0:
                    restored_best_epoch = int(best_epoch)

        if final_train_loss is None or final_train_metric is None:
            raise RuntimeError("Trainer.fit() completed without producing final train metrics.")

        result = {
            "best_train_loss": float(best_train_loss),
            "best_train_metric": float(best_train_metric),
            "final_train_loss": float(final_train_loss),
            "final_train_metric": float(final_train_metric),
            "best_val_loss": None if val_loader is None else float(best_val_loss),
            "best_val_metric": None if val_loader is None else float(best_val_metric),
            "final_val_loss": None if val_loader is None else final_val_loss,
            "final_val_metric": None if val_loader is None else final_val_metric,
            "best_epoch_by_val": None if val_loader is None else best_epoch_by_val,
            "restored_best_model": bool(restored_best_model),
            "restored_best_epoch": restored_best_epoch,
            "total_train_time_sec": float(total_time),
        }
        if final_custom_metrics:
            result["final_custom_metrics"] = final_custom_metrics
        if not self._is_silent():
            best_val_str = (
                f"best_val={self._fmt_scalar('val.metric', best_val_metric)}@{best_epoch_by_val}"
                if val_loader is not None and best_epoch_by_val is not None
                else "best_val=-"
            )
            done_line = (
                f"done | train_loss={self._fmt_scalar('loss', final_train_loss)} | "
                f"train_metric={self._fmt_scalar('metric', final_train_metric)} | "
                f"{best_val_str} | total={self._human_time(total_time)}"
            )
            if self._is_rich():
                self._console.print(f"[bold green]{done_line}[/]")
            else:
                print(done_line)
            if self.stop_reason:
                if self._is_rich():
                    self._console.print(f"[yellow]stop | {self.stop_reason}[/]")
                else:
                    print(f"stop | {self.stop_reason}")
        return result

    @torch.no_grad()
    def evaluate(self, loader, split: Optional[str] = None) -> Dict[str, Any]:
        self.model.eval()

        for probe in self.metric_probes:
            self._call_probe_hook(probe, "on_eval_start", self, split, self.state)

        total_n = 0
        total_loss = 0.0
        scalar_sums: dict[str, float] = {}

        for batch in loader:
            bs = infer_batch_size(batch)
            w = float(bs if bs > 0 else 1.0)
            loss, stats = compute_loss_and_stats(self.model, self.task, batch, self.device)
            if not isinstance(stats, Mapping):
                stats = {}
            stats_dict = dict(stats)

            total_loss += float(loss.item()) * w
            total_n += int(w)

            for key, value in stats_dict.items():
                if is_metric_payload_key(str(key)):
                    continue
                if isinstance(value, (int, float)):
                    scalar_sums[str(key)] = float(scalar_sums.get(str(key), 0.0) + (float(value) * w))

            for probe in self.metric_probes:
                self._call_probe_hook(
                    probe,
                    "on_eval_batch_end",
                    self,
                    split,
                    stats_dict,
                    int(max(1, bs)),
                    self.state,
                )

        denom = float(max(1, total_n))
        res: Dict[str, Any] = {"loss": float(total_loss / denom)}
        for key, weighted_sum in scalar_sums.items():
            if key == "loss":
                continue
            res[str(key)] = float(weighted_sum / denom)
        if "metric" not in res:
            if "acc" in res:
                res["metric"] = float(res["acc"])
            elif "agg" in res:
                res["metric"] = float(res["agg"])

        for probe in self.metric_probes:
            out = self._call_probe_hook(probe, "on_eval_end", self, split, self.state)
            if not isinstance(out, dict):
                continue
            for key, value in out.items():
                if isinstance(value, (int, float)):
                    res[str(key)] = float(value)

        self.log({"t": "eval", "split": split, **{k: float(v) for k, v in res.items() if isinstance(v, (int, float))}})
        if (not self._is_silent()) and (not self._in_fit):
            keys = self._display_metric_keys if self._display_metric_keys else ["metric"]
            fields = [f"loss={self._fmt_scalar('loss', res.get('loss'))}"]
            for key in keys:
                if key not in res:
                    continue
                fields.append(f"{key}={self._fmt_scalar(key, res.get(key))}")
            split_name = str(split or "eval")
            line = f"{split_name} | " + " ".join(fields)
            if self._is_rich():
                self._console.print(f"[cyan]{line}[/]")
            else:
                print(line)
        return res
