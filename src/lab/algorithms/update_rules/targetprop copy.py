# lab/algorithms/update_rules/targetprop.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from collections.abc import Mapping

from .base import UpdateRule
from .helpers import to_device, set_grad_, infer_activation_name, act_deriv_from_name


class TargetPropagation(UpdateRule):
    def __init__(
        self,
        fwd_lr=0.05,
        inv_lr=0.05,
        beta=1.0,
        noise_std=0.1,
        ste_heaviside=True,
        eps=1e-8,
        fwd_optimizer=None,
        inv_optimizer=None,
    ):
        super().__init__()
        self.fwd_lr = float(fwd_lr)
        self.inv_lr = float(inv_lr)
        self.beta = float(beta)
        self.noise_std = float(noise_std)
        self.ste_heaviside = bool(ste_heaviside)
        self.eps = float(eps)
        self.fwd_optimizer = fwd_optimizer
        self.inv_optimizer = inv_optimizer
        self.decoders = None  # list[torch.nn.Linear]

    def _ensure_decoders(self, linears, h_list):
        device = h_list[0].device
        dtype = h_list[0].dtype
        L = len(linears)

        if self.decoders is not None and len(self.decoders) == L:
            for d in self.decoders:
                d.to(device=device, dtype=dtype)
            return

        self.decoders = []
        for l in range(L):
            in_dim = linears[l].in_features
            out_dim = linears[l].out_features
            dec = torch.nn.Linear(out_dim, in_dim, bias=True).to(device=device, dtype=dtype)
            torch.nn.init.normal_(dec.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(dec.bias)
            self.decoders.append(dec)

            if self.inv_optimizer is not None:
                self.inv_optimizer.add_param_group({"params": list(dec.parameters())})

    def _train_inverse_maps(self, h_list, logits, linears):
        B = h_list[0].size(0)
        L = len(linears)

        if self.inv_optimizer is not None:
            self.inv_optimizer.zero_grad(set_to_none=True)

        for l in range(L):
            if l == L - 1:
                y_in = logits
                target = h_list[-1]
            else:
                y_in = h_list[l + 1]
                target = h_list[l]

            y_in = y_in + self.noise_std * torch.randn_like(y_in)
            dec = self.decoders[l]
            recon = dec(y_in)
            err = (target - recon)

            dW = (err.T @ y_in) / float(B)
            db = err.mean(dim=0)

            if self.inv_optimizer is None:
                dec.weight.add_(self.inv_lr * dW)
                dec.bias.add_(self.inv_lr * db)
            else:
                set_grad_(dec.weight, -dW)
                set_grad_(dec.bias, -db)

        if self.inv_optimizer is not None:
            self.inv_optimizer.step()

    def _output_target(self, logits, y_true):
        B = logits.size(0)
        p = F.softmax(logits, dim=1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, y_true.view(-1, 1), 1.0)
        dlogits = (p - onehot) / float(B)
        return logits - self.beta * dlogits

    def _propagate_targets(self, h_list, logits, t_logits, linears):
        L = len(linears)
        targets_h = [None] * len(h_list)
        t_l = t_logits

        for l in reversed(range(L)):
            dec = self.decoders[l]
            if l == L - 1:
                h_l = logits
                h_prev = h_list[-1]
            else:
                h_l = h_list[l + 1]
                h_prev = h_list[l]
            t_prev = h_prev + dec(t_l) - dec(h_l)
            targets_h[-1 if l == L - 1 else l] = t_prev
            t_l = t_prev

        return targets_h, t_logits

    def _update_forward_weights(self, linears, cache, targets_h, logits, t_logits, act_name: str):
        B = cache["acts"][0].size(0)
        L = len(linears)

        if self.fwd_optimizer is not None:
            self.fwd_optimizer.zero_grad(set_to_none=True)

        for l in range(L):
            layer = linears[l]
            x_l = cache["inputs"][l]
            z_l = cache["preacts"][l]

            if l == L - 1:
                delta = (t_logits - logits)
            else:
                h_out = cache["acts"][l + 1]
                t_h = targets_h[l + 1]
                deriv = act_deriv_from_name(z_l, act_name, ste_heaviside=self.ste_heaviside)
                delta = (t_h - h_out) * deriv

            dW = (delta.T @ x_l) / float(B)
            db = delta.mean(dim=0)

            if self.fwd_optimizer is None:
                layer.weight.add_(self.fwd_lr * dW)
                if layer.bias is not None:
                    layer.bias.add_(self.fwd_lr * db)
            else:
                set_grad_(layer.weight, -dW)
                if layer.bias is not None:
                    set_grad_(layer.bias, -db)

        if self.fwd_optimizer is not None:
            self.fwd_optimizer.step()

    @torch.no_grad()
    def train_step(self, model, task, batch, device, state=None):
        model.train()

        if isinstance(batch, Mapping):
            raise NotImplementedError("TargetPropagation: uniquement tuple batches (x,y) + model(x, return_cache=True).")

        x, y_true = to_device(batch, device)

        if "return_cache" not in model.forward.__code__.co_varnames:
            raise NotImplementedError("TargetPropagation expects model(x, return_cache=True).")
        if not hasattr(model, "linears"):
            raise NotImplementedError("TargetPropagation expects model.linears (MLPClassifier-style).")

        logits, cache = model(x, return_cache=True)
        linears = list(model.linears)
        act_name = infer_activation_name(model)

        h_list = cache["acts"]
        self._ensure_decoders(linears, h_list)
        self._train_inverse_maps(h_list, logits, linears)

        t_logits = self._output_target(logits, y_true)
        targets_h, t_logits = self._propagate_targets(h_list, logits, t_logits, linears)

        self._update_forward_weights(linears, cache, targets_h, logits, t_logits, act_name)

        loss = task.loss(logits, y_true)
        stats = {"loss": float(loss.item())}
        if hasattr(task, "metrics"):
            stats.update(task.metrics(logits, y_true))

        self.global_step += 1
        return stats
