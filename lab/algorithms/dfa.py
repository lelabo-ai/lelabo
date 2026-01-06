# algorithms/dfa.py
import math
import torch
import torch.nn.functional as F
from .base import Algorithm
from collections.abc import Mapping


class DirectFeedbackAlignment(Algorithm):
    """
    Direct Feedback Alignment (DFA) with an external optimizer.

    For each hidden layer l:
        delta_l = (delta_out @ B_l) * f'(z_l)
    where:
        delta_out = dL/dlogits  (cross-entropy gradient)
        B_l is fixed random matrix mapping output error directly to layer l.

    Then:
        grad_W_l = delta_l^T @ x_l / B
        grad_b_l = mean(delta_l)

    Output layer uses true delta_out:
        grad_W_L = delta_out^T @ x_L / B

    Supports MLP-style networks (2D Linear activations). For transformer-style
    dict batches, typically only the head Linear is updated (internal linears
    are often 3D and skipped).
    """

    def __init__(
        self,
        optimizer,
        feedback_scale=1.0,
        grad_clip=None,
        activation=None,    # override: "relu", "tanh", "sigmoid"
        grad_scale=1.0,     # scales all grads before optimizer step
        eps=1e-8,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.feedback_scale = float(feedback_scale)
        self.grad_clip = grad_clip
        self.activation = activation
        self.grad_scale = float(grad_scale)
        self.eps = float(eps)

        # For each hidden layer l (0..L-2), B_l has shape (C, out_l)
        # where C = num_classes (logits dim).
        self.B_mats = None
        self._cached_C = None  # remember output dim for shape checks

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

    def _ce_delta_logits(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # dL/dlogits for CE-softmax (averaged over batch)
        B = logits.size(0)
        probs = F.softmax(logits, dim=1)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, y_true.view(-1, 1), 1.0)
        return (probs - onehot) / float(B)

    def _ensure_feedback(self, out_dim: int, hidden_layers: list[torch.nn.Linear], device, dtype):
        """
        Create fixed random feedback matrices B_l for each hidden layer.
        B_l shape: (out_dim, out_l) so that delta_out @ B_l -> delta_l.
        """
        H = len(hidden_layers)
        if H <= 0:
            self.B_mats = []
            self._cached_C = out_dim
            return

        if (
            self.B_mats is not None
            and len(self.B_mats) == H
            and self._cached_C == out_dim
        ):
            ok = True
            for l, layer in enumerate(hidden_layers):
                if self.B_mats[l].shape != (out_dim, layer.out_features):
                    ok = False
                    break
            if ok:
                self.B_mats = [B.to(device=device, dtype=dtype) for B in self.B_mats]
                return

        self.B_mats = []
        self._cached_C = out_dim
        for layer in hidden_layers:
            out_l = layer.out_features
            # scale ~ 1/sqrt(out_dim) is a common stable choice
            scale = self.feedback_scale / math.sqrt(max(out_dim, 1))
            B = torch.randn(out_dim, out_l, device=device, dtype=dtype) * scale
            self.B_mats.append(B)

    # -------------------------
    # main API
    # -------------------------
    @torch.no_grad()
    def train_step(self, model, task, batch, device):
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # ---- capture (x, z) for each Linear layer ----
        layer_cache = []  # list of (x, z, module) in the order hooks fire
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

        # Can't proceed without labels or without captured layers
        if y_true is None or len(layer_cache) == 0:
            stats = {"loss": 0.0}
            self.global_step += 1
            return stats

        # Identify the "output layer" among cached linears:
        # best effort: choose the last cached Linear whose output matches logits shape
        out_idx = None
        for i in reversed(range(len(layer_cache))):
            if layer_cache[i][1].shape == logits.shape:
                out_idx = i
                break
        if out_idx is None:
            # fallback: assume last cached linear is output
            out_idx = len(layer_cache) - 1

        # Split layers into hidden + output
        # We'll only update linears up to out_idx (ignore any later cached linears, if weird architectures)
        layer_cache = layer_cache[: out_idx + 1]
        hidden_cache = layer_cache[:-1]
        out_cache = layer_cache[-1]

        hidden_layers = [m for (_, _, m) in hidden_cache]
        out_layer = out_cache[2]

        act_name = self._get_act_name(model)

        # ---- compute output delta ----
        delta_out = self._ce_delta_logits(logits, y_true)  # (B, C)
        C = logits.size(1)

        # ---- ensure direct feedback matrices for hidden layers ----
        self._ensure_feedback(C, hidden_layers, device=logits.device, dtype=logits.dtype)

        Bsz = logits.size(0)

        # ---- write grads for hidden layers (direct from output) ----
        for l, (x_l, z_l, layer) in enumerate(hidden_cache):
            # B_l: (C, out_l)
            Bmat = self.B_mats[l]
            # (B,C) @ (C,out_l) -> (B,out_l)
            delta_l = delta_out @ Bmat
            delta_l = delta_l * self._act_deriv(z_l, act_name)

            dW = (delta_l.T @ x_l) / float(Bsz)
            if layer.weight.grad is None:
                layer.weight.grad = torch.zeros_like(layer.weight)
            layer.weight.grad.copy_(self.grad_scale * dW)

            if layer.bias is not None:
                db = delta_l.mean(dim=0)
                if layer.bias.grad is None:
                    layer.bias.grad = torch.zeros_like(layer.bias)
                layer.bias.grad.copy_(self.grad_scale * db)

        # ---- output layer grads (true delta_out) ----
        x_L, z_L, layer_L = out_cache
        dW_L = (delta_out.T @ x_L) / float(Bsz)
        if layer_L.weight.grad is None:
            layer_L.weight.grad = torch.zeros_like(layer_L.weight)
        layer_L.weight.grad.copy_(self.grad_scale * dW_L)

        if layer_L.bias is not None:
            db_L = delta_out.mean(dim=0)
            if layer_L.bias.grad is None:
                layer_L.bias.grad = torch.zeros_like(layer_L.bias)
            layer_L.bias.grad.copy_(self.grad_scale * db_L)

        # ---- optional grad clipping & optimizer step ----
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        self.optimizer.step()

        # ---- stats ----
        loss = task.loss(logits, y_true)
        stats = {"loss": float(loss.item())}
        if hasattr(task, "metrics"):
            stats.update(task.metrics(logits, y_true))

        self.global_step += 1
        return stats
