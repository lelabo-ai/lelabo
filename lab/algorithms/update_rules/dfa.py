# lab/algorithms/update_rules/dfa.py
from __future__ import annotations

import torch
from collections.abc import Mapping

from .base import UpdateRule
from .helpers import (
    to_device,
    collect_linear_cache,
    split_hidden_and_output,
    infer_activation_name,
    act_deriv_from_name,
    ce_delta_logits,
    ensure_dfa_feedback,
    set_linear_grads_,
    optimizer_step,
)


class DirectFeedbackAlignment(UpdateRule):
    """
    DFA (no autograd): delta_l = (delta_out @ B_l) * f'(z_l)
    Support principal: MLP-style (model.linears + return_cache=True).
    """
    def __init__(self, optimizer, feedback_scale=1.0, grad_clip=None, activation=None, grad_scale=1.0):
        super().__init__()
        self.optimizer = optimizer
        self.feedback_scale = float(feedback_scale)
        self.grad_clip = grad_clip
        self.activation = activation
        self.grad_scale = float(grad_scale)
        self.B_mats = None
        self._cached_C = None

    @torch.no_grad()
    def train_step(self, model, task, batch, device):
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, Mapping):
            # Comme FA: on peut calculer une loss, mais DFA interne transformer sans cache fiable -> skip updates
            b = to_device(batch, device)
            outputs = model(**b)
            logits = outputs.logits
            y_true = b.get("labels", None)
            if y_true is None:
                self.global_step += 1
                return {"loss": 0.0}
            loss = task.loss(logits, y_true)
            self.global_step += 1
            return {"loss": float(loss.item())}

        x, y_true = to_device(batch, device)
        logits, layer_cache = collect_linear_cache(model, x)

        if y_true is None or len(layer_cache) == 0:
            self.global_step += 1
            return {"loss": 0.0}

        hidden_cache, out_cache = split_hidden_and_output(layer_cache, logits)
        if out_cache is None:
            self.global_step += 1
            return {"loss": 0.0}

        hidden_layers = [m for (_, _, m) in hidden_cache]
        act_name = infer_activation_name(model, self.activation)

        delta_out = ce_delta_logits(logits, y_true)  # (B,C)
        C = logits.size(1)

        self.B_mats = ensure_dfa_feedback(
            hidden_layers, out_dim=C, feedback_scale=self.feedback_scale,
            device=logits.device, dtype=logits.dtype, prev=self.B_mats
        )

        # ---- hidden layers ----
        for l, (x_l, z_l, layer) in enumerate(hidden_cache):
            Bmat = self.B_mats[l]  # (C, out_l)
            delta_l = (delta_out @ Bmat) * act_deriv_from_name(z_l, act_name)
            set_linear_grads_(layer, x_l, delta_l, scale=self.grad_scale)

        # ---- output layer (true delta_out) ----
        x_L, _z_L, layer_L = out_cache
        set_linear_grads_(layer_L, x_L, delta_out, scale=self.grad_scale)

        params = list(model.parameters())
        optimizer_step(self.optimizer, params_for_clip=params, grad_clip=self.grad_clip)

        loss = task.loss(logits, y_true)
        stats = {"loss": float(loss.item())}
        if hasattr(task, "metrics"):
            stats.update(task.metrics(logits, y_true))

        self.global_step += 1
        return stats
