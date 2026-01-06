# algorithms/feedback_alignment.py
import math
import torch
import torch.nn.functional as F
from .base import Algorithm
from collections.abc import Mapping


class FeedbackAlignment(Algorithm):
    """
    Feedback Alignment (FA) with an external optimizer.

    - Computes approximate gradients locally (no autograd)
    - Writes them into param.grad
    - Calls optimizer.step()

    Works best for MLPs (2D Linear activations).
    For transformer-style dict batches, it usually updates only the classifier head
    (since internal linears are often 3D and we skip those).
    """

    def __init__(
        self,
        optimizer,
        feedback_scale=1.0,
        grad_clip=None,
        activation=None,   # override: "relu", "tanh", "sigmoid"
        grad_scale=1.0,    # multiply all grads (useful if you want separate "algo lr")
        eps=1e-8,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.feedback_scale = float(feedback_scale)
        self.grad_clip = grad_clip
        self.activation = activation
        self.grad_scale = float(grad_scale)
        self.eps = float(eps)

        # B_l maps delta_{l+1} (out_{l+1}) -> delta_l (out_l), fixed random
        self.B_mats = None

    # -------------------------
    # helpers
    # -------------------------
    def _get_act_name(self, model) -> str:
        if self.activation is not None:
            return str(self.activation).lower()
        if hasattr(model, "activation"):
            return str(model.activation).lower()
        return "relu"

    def _act_deriv(self, z: torch.Tensor, act_name: str) -> torch.Tensor:
        if act_name == "relu":
            return (z > 0).to(z.dtype)
        if act_name == "tanh":
            h = torch.tanh(z)
            return 1.0 - h * h
        if act_name == "sigmoid":
            h = torch.sigmoid(z)
            return h * (1.0 - h)
        return (z > 0).to(z.dtype)

    def _ensure_feedback(self, linear_layers: list[torch.nn.Linear], device, dtype):
        L = len(linear_layers)
        if L < 2:
            self.B_mats = []
            return

        if self.B_mats is not None and len(self.B_mats) == (L - 1):
            ok = True
            for l in range(L - 1):
                out_next = linear_layers[l + 1].out_features
                out_cur = linear_layers[l].out_features
                if self.B_mats[l].shape != (out_next, out_cur):
                    ok = False
                    break
            if ok:
                self.B_mats = [B.to(device=device, dtype=dtype) for B in self.B_mats]
                return

        self.B_mats = []
        for l in range(L - 1):
            out_next = linear_layers[l + 1].out_features
            out_cur = linear_layers[l].out_features
            scale = self.feedback_scale / math.sqrt(max(out_next, 1))
            B = torch.randn(out_next, out_cur, device=device, dtype=dtype) * scale
            self.B_mats.append(B)

    def _ce_delta_logits(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # dL/dlogits for CE-softmax (already averaged by batch)
        B = logits.size(0)
        probs = F.softmax(logits, dim=1)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, y_true.view(-1, 1), 1.0)
        return (probs - onehot) / float(B)

    # -------------------------
    # main API
    # -------------------------
    @torch.no_grad()
    def train_step(self, model, task, batch, device):
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # ---- capture (x, z) for each Linear layer ----
        layer_cache = []  # list of (x, z) in the order hooks fire
        hooks = []

        def hook_fn(module, inputs, output):
            x = inputs[0]
            z = output
            # MLP-friendly only
            if x.dim() == 2 and z.dim() == 2:
                layer_cache.append((x.detach(), z.detach(), module))

        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                hooks.append(m.register_forward_hook(hook_fn))

        # ---- forward (tuple or dict batches) ----
        if isinstance(batch, Mapping):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            y_true = batch.get("labels", None)
        else:
            x, y_true = batch
            x, y_true = x.to(device), y_true.to(device)
            logits = model(x)

        for h in hooks:
            h.remove()

        # If no labels, can't compute CE delta (so we can't FA-train)
        if y_true is None or len(layer_cache) == 0:
            stats = {"loss": 0.0}
            self.global_step += 1
            return stats

        # linear layers we actually captured (already includes module ref)
        linear_layers = [m for (_, _, m) in layer_cache]
        act_name = self._get_act_name(model)

        # fixed random feedback
        self._ensure_feedback(linear_layers, device=logits.device, dtype=logits.dtype)

        # ---- deltas ----
        deltas = [None] * len(linear_layers)
        deltas[-1] = self._ce_delta_logits(logits, y_true)  # (B, C)

        # Backward with random feedback
        for l in reversed(range(len(linear_layers) - 1)):
            x_l, z_l, _ = layer_cache[l]
            delta_next = deltas[l + 1]           # (B, out_{l+1})
            Bmat = self.B_mats[l]                # (out_{l+1}, out_l)
            delta_l = delta_next @ Bmat          # (B, out_l)
            delta_l = delta_l * self._act_deriv(z_l, act_name)
            deltas[l] = delta_l

        # ---- write grads into .grad and optimizer.step() ----
        Bsz = logits.size(0)
        for l, (x_l, z_l, layer) in enumerate(layer_cache):
            delta_l = deltas[l]  # (B, out)

            # gradient estimate
            dW = (delta_l.T @ x_l) / float(Bsz)   # (out, in)
            if layer.weight.grad is None:
                layer.weight.grad = torch.zeros_like(layer.weight)
            layer.weight.grad.copy_(self.grad_scale * dW)

            if layer.bias is not None:
                db = delta_l.mean(dim=0)
                if layer.bias.grad is None:
                    layer.bias.grad = torch.zeros_like(layer.bias)
                layer.bias.grad.copy_(self.grad_scale * db)

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        self.optimizer.step()

        # ---- logging ----
        loss = task.loss(logits, y_true)
        stats = {"loss": float(loss.item())}
        if hasattr(task, "metrics"):
            stats.update(task.metrics(logits, y_true))

        self.global_step += 1
        return stats
