# algorithms/softhebb.py
import torch
import torch.nn.functional as F
from .base import UpdateRule
from collections.abc import Mapping


class SoftHebb(UpdateRule):
    """
    SoftHebb-style local learning:
      - Conv2d / Linear layers updated with:
            Δw_k = η_k * y_k * (x - u_k * w_k)
        where y = softmax(u / tau) (soft WTA)
      - Optional "soft anti-Hebbian": flip the sign for all non-winners.

    Integration notes:
      - Works with MLPs (Linear layers) and CNNs (Conv2d layers).
      - Uses forward hooks (like your KP).
      - Handles tuple batches (x,y) and dict batches (Transformers-style).
      - Optional supervised update for the head Linear that produces logits.

    This is not an exact reproduction of the paper’s full training protocol
    (greedy layer-wise + separate probe training), but it integrates cleanly
    into your per-batch training loop.
    """

    def __init__(
        self,
        learning_rate=0.05,
        tau=1.0,
        q=0.5,
        anti_hebb=True,
        # optional supervised head update (recommended with your current trainer)
        train_head=True,
        head_lr=0.05,
        eps=1e-8,
    ):
        super().__init__()
        self.lr = float(learning_rate)
        self.tau = float(tau)
        self.q = float(q)
        self.anti_hebb = bool(anti_hebb)

        self.train_head = bool(train_head)
        self.head_lr = float(head_lr)

        self.eps = float(eps)

    # -------------------------
    # helpers
    # -------------------------
    def _eta_per_unit_from_weight(self, w_2d: torch.Tensor) -> torch.Tensor:
        """
        w_2d: [K, D]
        returns eta_k: [K]
        """
        # norm per unit
        r = torch.norm(w_2d, dim=1)  # [K]
        # paper: η_k = η * (r_k - 1)^q
        # clamp for stability if r<1
        base = torch.clamp(r - 1.0, min=0.0)
        if self.q == 0.0:
            return torch.full_like(r, self.lr)
        return self.lr * (base + self.eps).pow(self.q)

    def _soft_wta(self, u: torch.Tensor) -> torch.Tensor:
        # softmax over "units/channels" dim=1
        return F.softmax(u / self.tau, dim=1)

    def _winner_sign(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B,K] or [B,K,L]
        returns S: same shape, +1 for winner, -1 for others
        """
        S = -torch.ones_like(y)
        if y.dim() == 2:
            idx = y.argmax(dim=1)  # [B]
            S[torch.arange(y.size(0), device=y.device), idx] = 1.0
        elif y.dim() == 3:
            # idx: [B,L], scatter along dim=1 (K)
            idx = y.argmax(dim=1)  # [B,L]
            S.scatter_(1, idx.unsqueeze(1), 1.0)
        return S

    def _update_linear_softhebb(self, layer: torch.nn.Linear, x: torch.Tensor, u: torch.Tensor):
        """
        x: [B, Din]
        u: [B, K]
        """
        B, Din = x.shape
        K = u.shape[1]

        W = layer.weight  # [K, Din]
        W2 = W  # already 2D

        y = self._soft_wta(u)  # [B,K]
        if self.anti_hebb:
            S = self._winner_sign(y)  # [B,K]
            y_eff = y * S
        else:
            y_eff = y

        # term1 = E[ y_eff_k * x ]
        term1 = (y_eff.T @ x) / float(B)  # [K,Din]

        # a_k = E[ y_eff_k * u_k ]
        a = (y_eff * u).mean(dim=0)  # [K]

        # Oja-like stabilization term: a_k * w_k
        dW = term1 - a[:, None] * W2  # [K,Din]

        eta_k = self._eta_per_unit_from_weight(W2)  # [K]
        layer.weight.add_(eta_k[:, None] * dW)

        if layer.bias is not None:
            # treat bias as weight from constant input 1:
            # Δb_k = η_k * y_eff_k * (1 - u_k * b_k)
            mean_y = y_eff.mean(dim=0)          # [K]
            db = mean_y - a * layer.bias        # [K]
            layer.bias.add_(eta_k * db)

    def _update_conv2d_softhebb(self, layer: torch.nn.Conv2d, x: torch.Tensor, u: torch.Tensor):
        """
        x: [B, Cin, Hin, Win]
        u: [B, Cout, Hout, Wout]
        """
        B, Cin, Hin, Win = x.shape
        Cout = u.shape[1]

        # unfold patches
        # patches: [B, D, L] where D=Cin*kH*kW, L=Hout*Wout
        patches = F.unfold(
            x,
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )

        D = patches.shape[1]
        L = patches.shape[2]

        u_ = u.reshape(B, Cout, L)  # [B,Cout,L]
        y = self._soft_wta(u_)      # [B,Cout,L]

        if self.anti_hebb:
            S = self._winner_sign(y)  # [B,Cout,L]
            y_eff = y * S
        else:
            y_eff = y

        # W_flat: [Cout, D]
        W = layer.weight
        W_flat = W.view(Cout, -1)

        # term1_k = E[ y_eff_k * patch ]
        # einsum: (B,Cout,L) x (B,D,L) -> (Cout,D)
        term1 = torch.einsum("bkl,bdl->kd", y_eff, patches) / float(B * L)

        # a_k = E[ y_eff_k * u_k ]
        a = (y_eff * u_).mean(dim=(0, 2))  # [Cout]

        dW_flat = term1 - a[:, None] * W_flat  # [Cout,D]

        eta_k = self._eta_per_unit_from_weight(W_flat)  # [Cout]

        # apply update
        W_flat.add_(eta_k[:, None] * dW_flat)
        layer.weight.copy_(W_flat.view_as(layer.weight))

        if layer.bias is not None:
            mean_y = y_eff.mean(dim=(0, 2))     # [Cout]
            db = mean_y - a * layer.bias        # [Cout]
            layer.bias.add_(eta_k * db)

    def _update_head_supervised(self, head: torch.nn.Linear, a: torch.Tensor, logits: torch.Tensor, y: torch.Tensor):
        """
        One-step SGD on cross-entropy for the head only, no autograd.
        a: [B, Din]    input to head
        logits: [B, C] output of head
        y: [B]
        """
        B = a.size(0)
        probs = F.softmax(logits, dim=1)  # [B,C]
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, y.view(-1, 1), 1.0)

        dlogits = (probs - onehot) / float(B)  # [B,C]
        dW = dlogits.T @ a                     # [C,Din]
        head.weight.add_(-self.head_lr * dW)

        if head.bias is not None:
            db = dlogits.sum(dim=0)            # [C]
            head.bias.add_(-self.head_lr * db)

    # -------------------------
    # main API
    # -------------------------
    @torch.no_grad()
    def train_step(self, model, task, batch, device):
        model.train()

        # ---- capture (module, x, u) for Conv2d and Linear ----
        layer_cache = []   # list of dicts: {"module":..., "x":..., "u":...}
        hooks = []

        def hook_fn(module, inputs, output):
            x = inputs[0]
            u = output
            # store only the cases we can handle robustly
            if isinstance(module, torch.nn.Conv2d) and x.dim() == 4 and u.dim() == 4:
                layer_cache.append({"module": module, "x": x.detach(), "u": u.detach()})
            elif isinstance(module, torch.nn.Linear) and x.dim() == 2 and u.dim() == 2:
                layer_cache.append({"module": module, "x": x.detach(), "u": u.detach()})

        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(m.register_forward_hook(hook_fn))

        # ---- forward (supports tuple or dict batches) ----
        y_true = None
        if isinstance(batch, Mapping):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            y_true = batch.get("labels", None)
        else:
            x, y_true = batch
            x, y_true = x.to(device), y_true.to(device)
            logits = model(x)

        # remove hooks ASAP
        for h in hooks:
            h.remove()

        # ---- identify head Linear that produced logits (if any) ----
        head_entry = None
        for entry in layer_cache:
            if isinstance(entry["module"], torch.nn.Linear):
                # match by tensor identity (data_ptr) when possible
                if entry["u"].data_ptr() == logits.data_ptr():
                    head_entry = entry
                    break
        if head_entry is None:
            # fallback: last Linear in cache whose output matches logits shape
            for entry in reversed(layer_cache):
                if isinstance(entry["module"], torch.nn.Linear) and entry["u"].shape == logits.shape:
                    head_entry = entry
                    break

        # ---- apply SoftHebb updates ----
        for entry in layer_cache:
            mod = entry["module"]
            x_m = entry["x"]
            u_m = entry["u"]

            if isinstance(mod, torch.nn.Conv2d):
                self._update_conv2d_softhebb(mod, x_m, u_m)

            elif isinstance(mod, torch.nn.Linear):
                is_head = (head_entry is not None and mod is head_entry["module"])
                if is_head and self.train_head and (y_true is not None):
                    # supervised update for classifier head
                    self._update_head_supervised(mod, x_m, logits, y_true)
                else:
                    # SoftHebb update (unsupervised) for Linear layers
                    self._update_linear_softhebb(mod, x_m, u_m)

        # ---- stats ----
        stats = {}
        if y_true is not None:
            # loss is just for logging; not used for weight updates here
            ce = task.loss(logits, y_true)
            stats["loss"] = float(ce.item())
            if hasattr(task, "metrics"):
                stats.update(task.metrics(logits, y_true))
        else:
            stats["loss"] = 0.0

        self.global_step += 1
        return stats
