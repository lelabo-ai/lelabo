# algorithms/targetprop.py
import torch
import torch.nn.functional as F
from .base import Algorithm
from collections.abc import Mapping


class TargetPropagation(Algorithm):
    """
    Difference Target Propagation (DTP) for MLPClassifier-style models.

    Supports passing optimizers:
      - fwd_optimizer: updates model forward weights (e.g. model.parameters())
      - inv_optimizer: updates inverse maps (decoders) created inside this algo

    If an optimizer is provided, we set .grad manually (no autograd) and call .step().
    Otherwise, we fallback to manual in-place SGD steps with fwd_lr / inv_lr.
    """

    def __init__(
        self,
        fwd_lr=0.05,
        inv_lr=0.05,
        beta=1.0,
        noise_std=0.1,
        ste_heaviside=True,
        eps=1e-8,
        # NEW:
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

    # -------------------------
    # utils
    # -------------------------
    @staticmethod
    def _set_grad(param: torch.nn.Parameter, grad: torch.Tensor):
        # make sure grad is same device/dtype
        g = grad.to(device=param.device, dtype=param.dtype)
        if param.grad is None:
            param.grad = g.clone()
        else:
            param.grad.copy_(g)

    def _infer_activation_name(self, model) -> str:
        if hasattr(model, "activation"):
            return str(model.activation).lower()
        return "relu"

    def _act_deriv(self, z: torch.Tensor, h: torch.Tensor, act_name: str) -> torch.Tensor:
        if act_name == "relu":
            return (z > 0).to(z.dtype)
        if act_name == "tanh":
            return 1.0 - h * h
        if act_name == "sigmoid":
            return h * (1.0 - h)
        if act_name == "heaviside":
            if self.ste_heaviside:
                return (z.abs() <= 1.0).to(z.dtype)  # simple STE band
            return torch.zeros_like(z)
        return (z > 0).to(z.dtype)

    def _ensure_decoders(self, linears, h_list):
        device = h_list[0].device
        dtype = h_list[0].dtype
        L = len(linears)

        if self.decoders is not None and len(self.decoders) == L:
            for d in self.decoders:
                if d.weight.device != device:
                    d.to(device=device, dtype=dtype)
            return

        self.decoders = []
        for l in range(L):
            in_dim = linears[l].in_features
            out_dim = linears[l].out_features
            dec = torch.nn.Linear(out_dim, in_dim, bias=True).to(device=device, dtype=dtype)

            torch.nn.init.normal_(dec.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(dec.bias)

            # we won't rely on autograd, but keep requires_grad=True by default
            self.decoders.append(dec)

            # If user provided inv_optimizer, add these params lazily
            if self.inv_optimizer is not None:
                self.inv_optimizer.add_param_group({"params": list(dec.parameters())})

    def _train_inverse_maps(self, h_list, logits, linears):
        """
        Train each decoder g_l by reconstruction:
          minimize 0.5 * || target - g_l(corrupted(input)) ||^2
        """
        B = h_list[0].size(0)
        L = len(linears)

        # optimizer path: accumulate grads, then step once
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
            err = (target - recon)  # [B, in_dim]

            # Update direction that we previously used:
            #   dW = (err^T @ y_in)/B, db = mean(err)
            # For an optimizer step (W -= lr*grad), set grad = -dW, -db
            dW = (err.T @ y_in) / float(B)
            db = err.mean(dim=0)

            if self.inv_optimizer is None:
                dec.weight.add_(self.inv_lr * dW)
                dec.bias.add_(self.inv_lr * db)
            else:
                self._set_grad(dec.weight, -dW)
                self._set_grad(dec.bias, -db)

        if self.inv_optimizer is not None:
            self.inv_optimizer.step()

    def _output_target(self, logits, y_true):
        B, C = logits.shape
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

            if l == L - 1:
                targets_h[-1] = t_prev
            else:
                targets_h[l] = t_prev

            t_l = t_prev

        return targets_h, t_logits

    def _update_forward_weights(self, linears, cache, h_list, targets_h, logits, t_logits, act_name: str):
        B = h_list[0].size(0)
        L = len(linears)

        preacts = cache["preacts"]  # length L, last is logits
        acts = cache["acts"]        # length L (input + hidden activations)

        if self.fwd_optimizer is not None:
            self.fwd_optimizer.zero_grad(set_to_none=True)

        for l in range(L):
            layer = linears[l]
            x_l = cache["inputs"][l]  # input to this layer
            z_l = preacts[l]

            if l == L - 1:
                delta = (t_logits - logits)          # [B, C]
            else:
                h_out = acts[l + 1]                  # post-activation
                t_h = targets_h[l + 1]
                deriv = self._act_deriv(z_l, h_out, act_name)
                delta = (t_h - h_out) * deriv

            dW = (delta.T @ x_l) / float(B)
            db = delta.mean(dim=0)

            if self.fwd_optimizer is None:
                layer.weight.add_(self.fwd_lr * dW)
                if layer.bias is not None:
                    layer.bias.add_(self.fwd_lr * db)
            else:
                self._set_grad(layer.weight, -dW)
                if layer.bias is not None:
                    self._set_grad(layer.bias, -db)

        if self.fwd_optimizer is not None:
            self.fwd_optimizer.step()

    # -------------------------
    # main API
    # -------------------------
    @torch.no_grad()
    def train_step(self, model, task, batch, device):
        model.train()

        # TP needs caches; we only support tuple batches here (like your MNIST/CIFAR MLP runs)
        if isinstance(batch, Mapping):
            batch = {k: v.to(device) for k, v in batch.items()}
            raise NotImplementedError(
                "TargetPropagation currently supports MLP-style models with return_cache=True (tuple batches). "
                "If you want GLUE/BERT, you’ll need the model to expose layerwise caches."
            )

        x, y_true = batch
        x, y_true = x.to(device), y_true.to(device)

        if "return_cache" not in model.forward.__code__.co_varnames:
            raise NotImplementedError("TargetPropagation expects model(x, return_cache=True).")

        logits, cache = model(x, return_cache=True)

        if not hasattr(model, "linears"):
            raise NotImplementedError("TargetPropagation is implemented for models with model.linears (MLPClassifier).")

        linears = list(model.linears)
        act_name = self._infer_activation_name(model)

        h_list = cache["acts"]  # [h0(input), h1, ..., h_{L-1}]

        # 1) ensure decoders
        self._ensure_decoders(linears, h_list)

        # 2) train inverse maps
        self._train_inverse_maps(h_list, logits, linears)

        # 3) output target
        t_logits = self._output_target(logits, y_true)

        # 4) propagate targets
        targets_h, t_logits = self._propagate_targets(h_list, logits, t_logits, linears)

        # 5) update forward weights
        self._update_forward_weights(linears, cache, h_list, targets_h, logits, t_logits, act_name)

        # stats
        loss = task.loss(logits, y_true)
        stats = {"loss": float(loss.item())}
        if hasattr(task, "metrics"):
            stats.update(task.metrics(logits, y_true))

        self.global_step += 1
        return stats
