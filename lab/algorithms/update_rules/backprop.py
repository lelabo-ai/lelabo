# lab/algorithms/update_rules/backprop.py
import torch
from .base import UpdateRule
from collections.abc import Mapping

class Backprop(UpdateRule):
    def __init__(self, optimizer, grad_clip=None):
        super().__init__()
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def train_step(self, model, task, batch, device):
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # --------- handle dict batches (GLUE / transformers) ----------
        if isinstance(batch, Mapping):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

            self.optimizer.step()

            stats = {"loss": float(loss.item())}

            if hasattr(outputs, "logits") and batch["labels"].dtype in (torch.int64, torch.int32, torch.int16):
                with torch.no_grad():
                    preds = outputs.logits.argmax(dim=-1)
                    acc = (preds == batch["labels"]).float().mean().item()
                stats["acc"] = acc

            self.global_step += 1
            return stats

        # --------- classic tuple batch (vision + RL-style) ----------
        x, y = batch
        x = x.to(device)

        # NEW: allow y to be dict-like for RL targets
        if isinstance(y, Mapping):
            y = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in y.items()}
        else:
            y = y.to(device)

        logits = model(x)
        loss = task.loss(logits, y)
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        self.optimizer.step()

        stats = {"loss": float(loss.item())}
        if hasattr(task, "metrics"):
            stats.update(task.metrics(logits, y))

        self.global_step += 1
        return stats
