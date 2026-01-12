# lab/algorithms/update_rules/feedback_alignment.py
from __future__ import annotations

import torch
from collections.abc import Mapping

from .base import UpdateRule
from .helpers import (
    to_device,
    collect_linear_cache,
    infer_activation_name,
    act_deriv_from_name,
    ce_delta_logits,
    ensure_fa_feedback,
    set_linear_grads_,
    optimizer_step,
)


class FeedbackAlignment(UpdateRule):
    """
    FA (no autograd): écrit des gradients estimés dans .grad puis optimizer.step().
    Support principal: MLP-style (model.linears + return_cache=True).
    Fallback possible via hooks si pas de cache.
    """
    def __init__(self, optimizer, feedback_scale=1.0, grad_clip=None, activation=None, grad_scale=1.0):
        super().__init__()
        self.optimizer = optimizer
        self.feedback_scale = float(feedback_scale)
        self.grad_clip = grad_clip
        self.activation = activation
        self.grad_scale = float(grad_scale)
        self.B_mats = None

    @torch.no_grad()
    def train_step(self, model, task, batch, device):
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # ---- forward & cache ----
        if isinstance(batch, Mapping):
            # FA nécessite labels; on supporte HF uniquement si labels existent + si on capte un Linear head (fallback hooks)
            b = to_device(batch, device)
            outputs = model(**b)
            logits = outputs.logits
            y_true = b.get("labels", None)
            # pour FA interne sur transformer, ça devient vite fragile -> si pas de cache, on update rien
            if y_true is None:
                self.global_step += 1
                return {"loss": 0.0}
            # pas de layer_cache fiable ici => on retourne loss uniquement
            loss = task.loss(logits, y_true)
            self.global_step += 1
            return {"loss": float(loss.item())}

        x, y_true = to_device(batch, device)
        logits, layer_cache = collect_linear_cache(model, x)

        if y_true is None or len(layer_cache) == 0:
            self.global_step += 1
            return {"loss": 0.0}

        linear_layers = [m for (_, _, m) in layer_cache]
        act_name = infer_activation_name(model, self.activation)

        # ---- feedback matrices ----
        self.B_mats = ensure_fa_feedback(
            linear_layers, self.feedback_scale, device=logits.device, dtype=logits.dtype, prev=self.B_mats
        )

        # ---- deltas ----
        deltas = [None] * len(linear_layers)
        deltas[-1] = ce_delta_logits(logits, y_true)  # (B,C)

        for l in reversed(range(len(linear_layers) - 1)):
            x_l, z_l, _ = layer_cache[l]
            delta_next = deltas[l + 1]          # (B, out_{l+1})
            Bmat = self.B_mats[l]               # (out_{l+1}, out_l)
            delta_l = (delta_next @ Bmat) * act_deriv_from_name(z_l, act_name)
            deltas[l] = delta_l

        # ---- write grads ----
        for (x_l, _z_l, layer), delta in zip(layer_cache, deltas):
            set_linear_grads_(layer, x_l, delta, scale=self.grad_scale)

        # ---- step ----
        params = list(model.parameters())
        optimizer_step(self.optimizer, params_for_clip=params, grad_clip=self.grad_clip)

        # ---- stats ----
        loss = task.loss(logits, y_true)
        stats = {"loss": float(loss.item())}
        if hasattr(task, "metrics"):
            stats.update(task.metrics(logits, y_true))

        self.global_step += 1
        return stats
