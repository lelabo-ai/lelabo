# algorithms/local_probe_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import UpdateRule

class LocalProbeBlocks(UpdateRule):
    def __init__(self, base_optimizer, probe_lr=1e-3):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.probe_lr = probe_lr
        self.probes = None
        self.probe_optim = None

    def on_train_start(self, model, task, device, state=None):
        if not hasattr(model, "local_blocks"):
            raise ValueError("Model has no local_blocks attribute; cannot use LocalProbeBlocks.")

        self._specs = model.local_blocks
        self.probes = nn.ModuleDict()
        self.probe_optim = None  # lazy init on first batch

    def train_step(self, model, task, batch, device, state=None):
        model.train()
        x, y = batch
        x, y = x.to(device), y.to(device)

        # We only use this forward for cache + logging → no need to build a graph
        with torch.no_grad():
            logits, cache = model(x, return_cache=True)

        # Lazy init probes when we see real shapes
        if self.probe_optim is None:
            for spec in self._specs:
                if spec.get("is_output", False):
                    continue
                name = spec["name"]
                rep = spec.get("rep", "identity")

                x_in = cache["block_inputs"][name].to(device)
                out = spec["module"](x_in)  # builds a graph for probe init? forces params grad, ok

                if out.dim() == 4 and rep == "gap":
                    dim = out.size(1)  # channels
                else:
                    dim = out.size(-1)

                self.probes[name] = nn.Linear(dim, task.num_classes).to(device)

            self.probe_optim = torch.optim.SGD(self.probes.parameters(), lr=self.probe_lr)

        # zero grads
        self.base_optimizer.zero_grad(set_to_none=True)
        self.probe_optim.zero_grad(set_to_none=True)

        # Local losses (these MUST track grads)
        for spec in self._specs:
            name = spec["name"]
            block = spec["module"]
            is_out = spec.get("is_output", False)
            rep = spec.get("rep", "identity")

            x_in = cache["block_inputs"][name].to(device)  # no grad history, fine
            out = block(x_in)                              # params require grad → graph exists

            if is_out:
                loss_local = task.loss(out, y)
            else:
                if out.dim() == 4 and rep == "gap":
                    feat = out.mean(dim=(2, 3))
                else:
                    feat = out
                probe_logits = self.probes[name](feat)
                loss_local = F.cross_entropy(probe_logits, y)

            loss_local.backward()

        self.base_optimizer.step()
        self.probe_optim.step()

        # Stats from the no-grad forward
        loss = task.loss(logits, y)
        stats = {"loss": float(loss.item())}
        stats.update(task.metrics(logits, y))
        self.global_step += 1
        return stats
