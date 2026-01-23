# lab/algorithms/update_rules/dni.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule


def _infer_num_classes(task: Any, y: torch.Tensor, logits: torch.Tensor) -> int:
    if hasattr(task, "num_classes"):
        try:
            return int(task.num_classes)
        except Exception:
            pass
    if logits is not None and torch.is_tensor(logits) and logits.dim() >= 2:
        return int(logits.size(-1))
    return int(y.max().item()) + 1


class _SGModel(nn.Module):
    """
    Synthetic gradient model M(h, c) -> grad wrt h.
    Default: linear. Optionally a 1-hidden-layer MLP.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            last = self.net[-1]
        else:
            self.net = nn.Linear(in_dim, out_dim)
            last = self.net

        # Start with zero synthetic gradients (paper-friendly init)
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNI(UpdateRule):
    """
    Decoupled Neural Interfaces (Synthetic Gradients), feed-forward setting.

    What this does (matching the paper + Supplement BP(λ)):
      - For each hidden activation h_i (between Linear layers), learn a model:
          ĝ_i = M_{i+1}(h_i[, y])
        that predicts dL/dh_i.
      - Train M_{i+1} by regressing to a *bootstrapped* target computed by
        backpropagating one step through the next layer:
          target_i = dL/dh_i = dL/du_{i+1} @ W_{i+1}
        (TD(0)-style target, analogous to the paper).
      - Update the main network using a mixed gradient (BP(λ)):
          g_i = λ * target_i + (1-λ) * scale * stopgrad(ĝ_i)

    Notes:
      - Minimal implementation focused on chains of nn.Linear.
      - Assumes a pointwise activation between linears (default ReLU).
      - Does not update non-Linear parameters (e.g. BatchNorm) — keep the backbone simple
        if you want closest behavior to the paper’s FCN experiments.
    """

    def __init__(
        self,
        lr: float = 3e-4,
        net_optim: str = "adamw",       # "adamw" | "sgd"
        net_weight_decay: float = 0.0,
        net_momentum: float = 0.9,      # only for SGD
        sg_lr: float = 3e-4,
        sg_optim: str = "adamw",        # "adamw" | "sgd"
        sg_weight_decay: float = 0.0,
        sg_hidden: int = 0,             # 0 => linear SG model (often best in the paper)
        condition_on_label: bool = False,  # cDNI
        lambda_mix: float = 0.0,        # BP(λ): 0 => pure synthetic, 1 => pure backprop through chain
        sg_scale: float = 1.0,          # multiply SG before consuming (stability knob)
        activation: str = "relu",       # "relu" | "tanh" | "identity"
        grad_clip: Optional[float] = None,
    ):
        super().__init__()
        self.lr = float(lr)
        self.net_optim = str(net_optim).lower()
        self.net_weight_decay = float(net_weight_decay)
        self.net_momentum = float(net_momentum)

        self.sg_lr = float(sg_lr)
        self.sg_optim = str(sg_optim).lower()
        self.sg_weight_decay = float(sg_weight_decay)
        self.sg_hidden = int(sg_hidden)

        self.condition_on_label = bool(condition_on_label)
        self.lambda_mix = float(lambda_mix)
        self.sg_scale = float(sg_scale)

        self.activation = str(activation).lower()
        self.grad_clip = grad_clip

        self._net_optimizer: Optional[torch.optim.Optimizer] = None
        self._sg_optimizer: Optional[torch.optim.Optimizer] = None

        # lazily created once we see shapes
        self._sg_models: nn.ModuleList = nn.ModuleList()
        self._built_for: Optional[Tuple[int, ...]] = None
        self._num_classes: Optional[int] = None

    # ---------------- activation utilities ----------------

    def _act(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(u)
        if self.activation == "tanh":
            return torch.tanh(u)
        return u  # identity

    def _act_grad(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return (u > 0).to(u.dtype)
        if self.activation == "tanh":
            t = torch.tanh(u)
            return 1.0 - t * t
        return torch.ones_like(u)

    # ---------------- capture in execution order ----------------

    def _capture_linear_xu_hooks(self, model: nn.Module):
        """
        Capture Linear layers in *execution order*:
          x: input to Linear  [B, Din]  (this is h_{i-1})
          u: output of Linear [B, Dout] (pre-activation)
        """
        cache: List[Dict[str, Any]] = []
        hooks: List[Any] = []

        def hook_fn(module, inputs, output):
            x = inputs[0]
            u = output
            if isinstance(module, nn.Linear) and x.dim() == 2 and u.dim() == 2:
                cache.append({"module": module, "x": x.detach(), "u": u.detach()})

        for m in model.modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(hook_fn))

        return cache, hooks

    # ---------------- optimizers ----------------

    def _ensure_optimizers(self, model: nn.Module):
        if self._net_optimizer is None:
            if self.net_optim == "sgd":
                self._net_optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.lr,
                    momentum=self.net_momentum,
                    weight_decay=self.net_weight_decay,
                )
            else:
                self._net_optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.lr,
                    weight_decay=self.net_weight_decay,
                )

        if self._sg_optimizer is None:
            if self.sg_optim == "sgd":
                self._sg_optimizer = torch.optim.SGD(
                    self._sg_models.parameters(),
                    lr=self.sg_lr,
                    momentum=0.0,
                    weight_decay=self.sg_weight_decay,
                )
            else:
                self._sg_optimizer = torch.optim.AdamW(
                    self._sg_models.parameters(),
                    lr=self.sg_lr,
                    weight_decay=self.sg_weight_decay,
                )

    def _onehot(self, y: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(y.long(), num_classes=num_classes).to(dtype=torch.float32, device=y.device)

    def _maybe_build_sg_models(
        self,
        task: Any,
        y: torch.Tensor,
        linear_cache: List[Dict[str, Any]],
        logits: torch.Tensor,
        device: str,
    ):
        """
        Build one SG model per hidden activation h_i (i=0..N-2),
        i.e. len(linears)-1 models.
        """
        if len(linear_cache) < 2:
            self._sg_models = nn.ModuleList()
            self._built_for = tuple()
            return

        # signature for rebuild detection: (in0,out0,in1,out1,...)
        sig: List[int] = []
        for e in linear_cache:
            m: nn.Linear = e["module"]
            sig.extend([int(m.in_features), int(m.out_features)])
        signature = tuple(sig)

        if self._built_for == signature and len(self._sg_models) == (len(linear_cache) - 1):
            return

        self._built_for = signature
        self._sg_models = nn.ModuleList()

        num_classes = _infer_num_classes(task, y, logits)
        self._num_classes = num_classes

        for i in range(len(linear_cache) - 1):
            dim_hi = int(linear_cache[i]["u"].size(1))  # h_i dim = out_features of layer i
            in_dim = dim_hi + (num_classes if self.condition_on_label else 0)
            self._sg_models.append(_SGModel(in_dim=in_dim, out_dim=dim_hi, hidden_dim=self.sg_hidden))

        self._sg_models.to(device)
        self._sg_optimizer = None  # force rebuild because params changed

    # ---------------- main API ----------------

    def train_step(self, model: nn.Module, task: Any, batch, device: str) -> dict:
        model.train()

        if isinstance(batch, Mapping):
            raise NotImplementedError("DNI: Mapping/HF batches not supported in this minimal version.")

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        # 1) forward (no autograd) + capture Linear x/u
        linear_cache, hooks = self._capture_linear_xu_hooks(model)
        with torch.no_grad():
            out = model(x)
        for h in hooks:
            h.remove()

        logits = out["logits"] if isinstance(out, Mapping) else out
        # prefer last Linear's u if it matches logits shape (typical classifier head)
        if linear_cache and torch.is_tensor(logits) and linear_cache[-1]["u"].shape == logits.shape:
            logits = linear_cache[-1]["u"]

        # 2) build SG models lazily + optimizers
        self._maybe_build_sg_models(task, y, linear_cache, logits, device)
        self._ensure_optimizers(model)

        # 3) true gradient at output only (anchors the chain)
        logits_var = logits.detach().requires_grad_(True)
        loss_var = task.loss(logits_var, y)
        (g_out,) = torch.autograd.grad(loss_var, logits_var)
        g_out = g_out.detach()

        # logging stats
        with torch.no_grad():
            stats: Dict[str, float] = {"loss": float(loss_var.item())}
            if hasattr(task, "metrics"):
                try:
                    stats.update(task.metrics(logits, y))
                except Exception:
                    pass

        if len(linear_cache) == 0:
            self.global_step += 1
            return stats

        # 4) unpack cache
        N = len(linear_cache)
        u_list = [e["u"] for e in linear_cache]        # u_i
        x_list = [e["x"] for e in linear_cache]        # h_{i-1}
        lin_modules = [e["module"] for e in linear_cache]

        # h_i (post-activation) for i < N-1, and logits for last
        h_list: List[torch.Tensor] = []
        for i in range(N):
            if i < N - 1:
                h_list.append(self._act(u_list[i]))
            else:
                h_list.append(u_list[i])

        # 5) backward recursion: compute boundary grads + SG regression losses
        g_list: List[Optional[torch.Tensor]] = [None] * N
        g_list[N - 1] = g_out

        delta_u: List[Optional[torch.Tensor]] = [None] * N
        delta_u[N - 1] = g_out

        sg_total_loss = torch.tensor(0.0, device=device)
        onehot = None
        if self.condition_on_label:
            onehot = self._onehot(y, int(self._num_classes or logits.size(-1)))

        for i in range(N - 2, -1, -1):
            j = i + 1  # next layer

            # delta_u_j = dL/du_j
            if j < N - 1:
                delta_u_j = g_list[j] * self._act_grad(u_list[j])
            else:
                delta_u_j = g_list[j]
            delta_u[j] = delta_u_j

            # target_i = dL/dh_i = delta_u_j @ W_j
            Wj = lin_modules[j].weight.detach()   # [Dout_j, Din_j]
            target = delta_u_j @ Wj               # [B, Din_j] == [B, dim(h_i)]

            # SG prediction + BP(lambda) mixing
            if len(self._sg_models) > 0:
                hi = h_list[i].detach()
                if self.condition_on_label:
                    inp = torch.cat([hi, onehot], dim=1)
                else:
                    inp = hi
                sg_pred = self._sg_models[i](inp)

                sg_total_loss = sg_total_loss + F.mse_loss(sg_pred, target.detach())

                g_i = (self.lambda_mix * target) + ((1.0 - self.lambda_mix) * (self.sg_scale * sg_pred.detach()))
            else:
                g_i = target

            g_list[i] = g_i

        # delta_u_0
        if N > 1:
            delta_u[0] = g_list[0] * self._act_grad(u_list[0])
        else:
            delta_u[0] = g_list[0]

        # 6) apply updates to main net (manual grads -> optimizer.step)
        self._net_optimizer.zero_grad(set_to_none=True)

        B = float(x_list[0].size(0))
        for i in range(N):
            lin: nn.Linear = lin_modules[i]
            du = delta_u[i]
            if du is None:
                continue

            grad_w = (du.T @ x_list[i]) / B
            lin.weight.grad = grad_w

            if lin.bias is not None:
                lin.bias.grad = du.mean(dim=0)

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        self._net_optimizer.step()

        # 7) update SG models
        if self._sg_optimizer is not None and len(self._sg_models) > 0:
            self._sg_optimizer.zero_grad(set_to_none=True)
            sg_total_loss.backward()
            self._sg_optimizer.step()
            stats["sg_loss"] = float(sg_total_loss.detach().item())

        self.global_step += 1
        return stats
