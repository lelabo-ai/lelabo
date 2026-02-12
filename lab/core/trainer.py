# lab/core/trainer.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from .callbacks import Callback
from .utils.logger import RunLogger
from .batch import infer_batch_size
from .steps import compute_loss_and_stats

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class Trainer:
    def __init__(
        self,
        model,
        task,
        learner,
        device: str = "cpu",
        input_noise_training: float = 0.0,
        verbose: bool = False,
        callbacks: Optional[list[Callback]] = None,
        logger: Optional[RunLogger] = None,
    ):
        self.model = model.to(device)
        self.task = task
        self.input_noise_training = float(input_noise_training)

        self.learner = learner
        self.algorithm = learner  # compat

        self.device = device
        self.verbose = verbose
        self.logger = logger or RunLogger(run_dir=None)
        self.callbacks = callbacks or []
        self.stop_training = False
        self.stop_reason = None

    def log(self, record: Dict[str, Any]) -> None:
        self.logger.log(record)

    def _maybe_sync_cuda(self) -> None:
        if isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def fit(self, train_loader, epochs: int = 10, show_progress: bool = True, val_loader=None) -> Dict[str, Any]:
        best_val_metric = float("-inf")
        best_val_loss = float("inf")
        best_epoch_by_val = None

        best_train_metric = float("-inf")
        best_train_loss = float("inf")

        final_val_metric = None
        final_val_loss = None

        for cb in self.callbacks:
            cb.on_train_start(self)

        self.learner.on_train_start(self.model, self.task, self.device)

        total_loss_sum = 0.0
        total_metric_sum = 0.0
        total_samples = 0
        total_batches = 0

        epoch_times: list[float] = []
        epoch_samples_per_sec: list[float] = []

        self._maybe_sync_cuda()
        train_start = time.perf_counter()

        last_epoch_ran = 0

        for ep in range(1, epochs + 1):
            last_epoch_ran = ep
            self.model.train()

            self._maybe_sync_cuda()
            ep_start = time.perf_counter()

            ep_loss_sum = 0.0
            ep_metric_sum = 0.0
            ep_samples = 0
            ep_batches = 0

            iterator = train_loader
            use_bar = show_progress and (tqdm is not None)
            if use_bar:
                iterator = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)

            for batch in iterator:
                bs = infer_batch_size(batch)
                ep_samples += bs

                if self.input_noise_training > 0.0 and isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x, y = batch
                    if torch.is_tensor(x) and x.is_floating_point():
                        x = x.to(self.device)
                        x = x + torch.randn_like(x) * self.input_noise_training
                        batch = (x, y)

                stats = self.learner.train_step(self.model, self.task, batch, self.device, ep)

                loss = float(stats.get("loss", 0.0))
                metric = float(stats.get("acc", stats.get("agg", 0.0)))

                w = float(bs if bs > 0 else 1)
                ep_loss_sum += loss * w
                ep_metric_sum += metric * w
                ep_batches += 1

                if use_bar:
                    denom = max(1.0, float(ep_samples))
                    iterator.set_postfix(loss=(ep_loss_sum / denom), metric=(ep_metric_sum / denom))

            self._maybe_sync_cuda()
            ep_time = time.perf_counter() - ep_start
            epoch_times.append(float(ep_time))

            denom = max(1.0, float(ep_samples))
            mean_loss = ep_loss_sum / denom
            mean_metric = ep_metric_sum / denom

            best_train_metric = max(best_train_metric, mean_metric)
            best_train_loss = min(best_train_loss, mean_loss)

            samples_per_sec = (ep_samples / ep_time) if ep_time > 0 else 0.0
            batches_per_sec = (ep_batches / ep_time) if ep_time > 0 else 0.0
            epoch_samples_per_sec.append(float(samples_per_sec))

            total_loss_sum += ep_loss_sum
            total_metric_sum += ep_metric_sum
            total_samples += ep_samples
            total_batches += ep_batches

            self.log({"t": "train", "epoch": ep, "loss": float(mean_loss), "metric": float(mean_metric)})

            logs: Dict[str, float] = {"train.loss": float(mean_loss), "train.metric": float(mean_metric)}

            if val_loader is not None:
                val_res = self.evaluate(val_loader, split="val")
                vloss = float(val_res.get("loss", 0.0))
                vmetric = float(val_res.get("acc", val_res.get("agg", val_res.get("metric", 0.0))))

                logs["val.loss"] = vloss
                logs["val.metric"] = vmetric

                if vmetric > best_val_metric:
                    best_val_metric = vmetric
                    best_val_loss = vloss
                    best_epoch_by_val = ep

                final_val_metric = vmetric
                final_val_loss = vloss

            for cb in self.callbacks:
                cb.on_epoch_end(self, ep, logs)

            if self.stop_training:
                break

            if self.verbose:
                print(f"Epoch {ep}/{epochs} | train_loss={mean_loss:.4f} | train_metric={mean_metric:.4f}")

        self._maybe_sync_cuda()
        total_time = time.perf_counter() - train_start

        final_train_loss = float(total_loss_sum / max(1.0, float(total_samples)))
        final_train_metric = float(total_metric_sum / max(1.0, float(total_samples)))

        for cb in self.callbacks:
            cb.on_train_end(self, {"last_epoch": float(last_epoch_ran)})

        return {
            "best_train_loss": float(best_train_loss),
            "best_train_metric": float(best_train_metric),
            "final_train_loss": final_train_loss,
            "final_train_metric": final_train_metric,
            "best_val_loss": None if val_loader is None else float(best_val_loss),
            "best_val_metric": None if val_loader is None else float(best_val_metric),
            "final_val_loss": None if val_loader is None else final_val_loss,
            "final_val_metric": None if val_loader is None else final_val_metric,
            "best_epoch_by_val": None if val_loader is None else best_epoch_by_val,
            "total_train_time_sec": float(total_time),
        }

    @torch.no_grad()
    def evaluate(self, loader, split: Optional[str] = None) -> Dict[str, Any]:
        self.model.eval()

        if hasattr(self.task, "evaluate"):
            res = self.task.evaluate(self.model, loader, self.device)
        else:
            total_n = 0
            total_loss = 0.0
            total_acc = 0.0

            for batch in loader:
                bs = infer_batch_size(batch)
                loss, stats = compute_loss_and_stats(self.model, self.task, batch, self.device)
                w = float(bs if bs > 0 else 1)
                total_loss += float(loss.item()) * w
                if "acc" in stats:
                    total_acc += float(stats["acc"]) * w
                total_n += int(bs if bs > 0 else 1)

            res = {"loss": float(total_loss / max(1, total_n)), "acc": float(total_acc / max(1, total_n)), "metric": float(total_acc / max(1, total_n))}

        self.log({"t": "eval", "split": split, **{k: float(v) for k, v in res.items() if isinstance(v, (int, float))}})
        if self.verbose:
            print(f"Eval {split or ''} | loss={res.get('loss', 0.0):.4f} | acc={res.get('acc', 0.0):.4f}")
        return res
