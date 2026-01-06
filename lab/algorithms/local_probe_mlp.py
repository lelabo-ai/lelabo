import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Algorithm

class LocalProbeMLP(Algorithm):
    """
    Local Probe Learning for MLP.
    Probes are updated using the SAME base optimizer (Adam/SGD/AdamW/etc),
    via optimizer param groups (optionally with probe_lr).
    """
    def __init__(self, base_optimizer, probe_lr=None, weight_decay=0.0):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.probe_lr = probe_lr  # if None, probes use optimizer's default lr
        self.weight_decay = weight_decay
        self.probes = None
        self._probes_registered = False

    def on_train_start(self, model, task, device):
        n_hidden_layers = len(model.linears) - 1
        self.probes = nn.ModuleList([
            nn.Linear(model.hidden_dim, task.num_classes).to(device)
            for _ in range(n_hidden_layers)
        ])

        # Register probe params into the SAME optimizer (new param group)
        # so they use the same update rule (Adam/SGD/...)
        if not self._probes_registered:
            params = list(self.probes.parameters())
            if len(params) > 0:
                group = {"params": params}
                if self.probe_lr is not None:
                    group["lr"] = self.probe_lr
                # If you want probes to have different weight decay, you could set it here too.
                self.base_optimizer.add_param_group(group)
            self._probes_registered = True

    def train_step(self, model, task, batch, device):
        model.train()
        x, y = batch
        x, y = x.to(device), y.to(device)
        bs = x.shape[0]
        num_classes = task.num_classes

        # forward to get cache
        logits, cache = model(x, return_cache=True)

        # one-hot targets
        y_oh = F.one_hot(y, num_classes=num_classes).to(dtype=logits.dtype)

        # IMPORTANT: now we also need grads for probe params, so DO NOT use @torch.no_grad()
        self.base_optimizer.zero_grad(set_to_none=True)

        n_hidden_layers = len(model.linears) - 1
        for l in range(n_hidden_layers):
            h = cache["acts"][l + 1]          # [bs, H]
            z = cache["preacts"][l]           # [bs, H]
            x_in = cache["inputs"][l]         # [bs, Din]

            probe = self.probes[l]

            # ---- Probe forward (no autograd needed; we write grads manually) ----
            probe_logits = h @ probe.weight.t() + probe.bias  # [bs, C]
            p = torch.softmax(probe_logits, dim=1)

            d_probe_logits = (p - y_oh) / bs                  # [bs, C]

            # ---- Probe grads (write into .grad so optimizer updates them) ----
            grad_PW = d_probe_logits.t() @ h                  # [C, H] matches probe.weight shape
            grad_Pb = d_probe_logits.sum(0)                   # [C] matches probe.bias shape

            probe.weight.grad = grad_PW.contiguous()
            probe.bias.grad = grad_Pb.contiguous()

            # ---- Local gradient for network layer l (manual) ----
            d_h = d_probe_logits @ probe.weight               # [bs, H]
            d_z = d_h * model.act_deriv_from_preact(z)        # [bs, H]

            grad_W = (d_z.t() @ x_in) / bs                    # [H, Din] matches lin.weight shape
            grad_b = d_z.mean(0)                              # [H] matches lin.bias shape

            lin = model.linears[l]
            lin.weight.grad = grad_W.contiguous()
            lin.bias.grad = grad_b.contiguous()

        # ---- Output layer grads (manual CE) ----
        probs = torch.softmax(logits, dim=1)
        d_logits = (probs - y_oh) / bs                        # [bs, C]
        a_last = cache["inputs"][-1]                          # [bs, H]

        grad_W_out = d_logits.t() @ a_last                    # [C, H] = out.weight shape
        grad_b_out = d_logits.sum(0)                          # [C]

        out_lin = model.linears[-1]
        out_lin.weight.grad = grad_W_out.contiguous()
        out_lin.bias.grad = grad_b_out.contiguous()

        # Optional extra decoupled weight decay (careful if using AdamW already)
        if self.weight_decay > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = p.grad + self.weight_decay * p.data

        # Single optimizer step updates BOTH base params and probes
        self.base_optimizer.step()

        loss = task.loss(logits, y)
        out = {"loss": float(loss.item())}
        out.update(task.metrics(logits, y))
        self.global_step += 1
        return out
