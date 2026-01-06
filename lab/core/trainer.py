# core/trainer.py
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Trainer:
    def __init__(
        self,
        model,
        task,
        algorithm,
        device: str = "cpu",
        input_noise_training: float = 0.0,
        verbose: bool = False,
        run_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.task = task
        self.input_noise_training = float(input_noise_training)
        self.algorithm = algorithm
        self.device = device
        self.verbose = verbose

        self.run_dir = Path(run_dir) if run_dir else None
        self.metrics_path = (self.run_dir / "metrics.jsonl") if self.run_dir else None

    def log(self, record: Dict[str, Any]) -> None:
        if self.metrics_path is None:
            return
        _append_jsonl(self.metrics_path, record)

    def fit(self, train_loader, epochs: int = 10, show_progress: bool = True) -> Dict[str, Any]:
        self.algorithm.on_train_start(self.model, self.task, self.device)

        best_train_acc = float("-inf")
        best_train_loss = float("inf")

        total_sum_loss = 0.0
        total_sum_acc = 0.0
        total_batches = 0
        total_samples = 0

        epoch_times: list[float] = []
        epoch_samples_per_sec: list[float] = []

        train_start = time.perf_counter()

        for ep in range(1, epochs + 1):
            self.model.train()

            ep_start = time.perf_counter()

            sum_loss = 0.0
            sum_acc = 0.0
            n_batches = 0
            n_samples = 0

            iterator = train_loader
            use_bar = show_progress and (tqdm is not None)
            if use_bar:
                iterator = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)

            for batch in iterator:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                # count samples
                try:
                    bs = int(x.size(0))
                except Exception:
                    bs = 0
                n_samples += bs

                if self.input_noise_training > 0.0:
                    x = x + torch.randn_like(x) * self.input_noise_training

                stats = self.algorithm.train_step(self.model, self.task, (x, y), self.device)

                loss = float(stats.get("loss", 0.0))
                acc = float(stats.get("acc", 0.0))

                sum_loss += loss
                sum_acc += acc
                n_batches += 1

                if use_bar:
                    iterator.set_postfix(loss=sum_loss / max(1, n_batches), acc=sum_acc / max(1, n_batches))

            ep_time = time.perf_counter() - ep_start
            epoch_times.append(float(ep_time))

            mean_loss = sum_loss / max(1, n_batches)
            mean_acc = sum_acc / max(1, n_batches)

            best_train_acc = max(best_train_acc, mean_acc)
            best_train_loss = min(best_train_loss, mean_loss)

            # Throughput
            samples_per_sec = (n_samples / ep_time) if ep_time > 0 else 0.0
            batches_per_sec = (n_batches / ep_time) if ep_time > 0 else 0.0
            epoch_samples_per_sec.append(float(samples_per_sec))

            # Global accumulators (useful for global throughput)
            total_sum_loss += sum_loss
            total_sum_acc += sum_acc
            total_batches += n_batches
            total_samples += n_samples

            # Log epoch train summary
            self.log(
                {
                    "t": "train",
                    "epoch": ep,
                    "loss": float(mean_loss),
                    "acc": float(mean_acc),
                    "epoch_time_sec": float(ep_time),
                    "samples": int(n_samples),
                    "batches": int(n_batches),
                    "samples_per_sec": float(samples_per_sec),
                    "batches_per_sec": float(batches_per_sec),
                }
            )

            if self.verbose:
                print(
                    f"Epoch {ep}/{epochs} | "
                    f"train_loss={mean_loss:.4f} | train_acc={mean_acc*100:.2f}% | "
                    f"time={ep_time:.2f}s | {samples_per_sec:.1f} samples/s"
                )

        total_time = time.perf_counter() - train_start

        # “Overall” throughput across the whole training
        overall_samples_per_sec = (total_samples / total_time) if total_time > 0 else 0.0
        overall_batches_per_sec = (total_batches / total_time) if total_time > 0 else 0.0

        # Final epoch stats (safe if epochs==0 not expected, but keep sane)
        final_train_loss = float((total_sum_loss / max(1, total_batches)))
        final_train_acc = float((total_sum_acc / max(1, total_batches)))

        return {
            "best_train_loss": float(best_train_loss),
            "best_train_acc": float(best_train_acc),
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "total_train_time_sec": float(total_time),
            "overall_samples_per_sec": float(overall_samples_per_sec),
            "overall_batches_per_sec": float(overall_batches_per_sec),
            "mean_epoch_time_sec": float(sum(epoch_times) / max(1, len(epoch_times))),
            "mean_epoch_samples_per_sec": float(sum(epoch_samples_per_sec) / max(1, len(epoch_samples_per_sec))),
        }

    @torch.no_grad()
    def evaluate(self, loader, split: Optional[str] = None) -> Dict[str, Any]:
        self.model.eval()

        eval_start = time.perf_counter()

        if hasattr(self.task, "evaluate"):
            res = self.task.evaluate(self.model, loader, self.device)
        else:
            total_acc = 0.0
            total_n = 0
            total_loss = 0.0

            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.task.loss(logits, y)
                acc = self.task.metrics(logits, y)["acc"]

                n = x.size(0)
                total_loss += loss.item() * n
                total_acc += acc * n
                total_n += n

            res = {"loss": total_loss / total_n, "acc": total_acc / total_n}

        eval_time = time.perf_counter() - eval_start

        # Log eval (include timing)
        payload: Dict[str, Any] = {"t": "eval", "eval_time_sec": float(eval_time)}
        if split is not None:
            payload["split"] = split

        # keep numeric metrics
        for k, v in res.items():
            if isinstance(v, (int, float)):
                payload[k] = float(v)

        self.log(payload)
        return res
