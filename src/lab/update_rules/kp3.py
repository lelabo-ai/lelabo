# lab/update_rules/softhebb.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .base import UpdateRule
from .registry import UpdateRuleContext, register_update_rule
from ..core.batch import to_device
from ..core.steps import maybe_accuracy_from_logits, metric_payload_from_outputs


class KP3(UpdateRule):
    """
    KP3: Hybrid rule with *local supervised Sanger/GHA-like learning* in the backbone.

    - Output blocks (bspec.is_output=True): optional standard backprop (heads-only optimizer).
      Set head_optim="none" or head_lr<=0 to disable fully.

    - Intermediate nn.Linear blocks: updated with a local supervised Hebbian rule that combines:
        * within-class residuals (compact)
        * between-class mean offsets (separate)
      and uses a Sanger/GHA term to prevent collapse (neurons learn different directions).
      No backward() is used for these updates.

    - Intermediate Conv2d blocks: pooled approximation:
        * x_eff = GAP(xin)                             [B, Cin]
        * maintains class/global means of x_eff
        * updates averaged conv filters (mean over kh,kw) using the same Sanger rule,
          then broadcasts the delta back to the kernel weights.
    """

    def __init__(
        self,
        # heads (backprop)
        head_lr: float = 3e-4,
        head_optim: str = "ano",  # "adamw" | "sgd" | "ano" | "none"
        head_weight_decay: float = 0.0,
        # local supervised Sanger
        local_lr: float = 3e-4,
        alpha_between: float = 1.0,   # separation strength
        beta_within: float = 1.0,     # compactness strength
        mean_momentum: float = 0.05,  # EMA for means
        sanger_diag_eps: float = 0.0, # optional: add eps on diag of tril matrix
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = float(eps)

        # --- heads (global supervised)
        self.head_lr = float(head_lr)
        self.head_optim = str(head_optim).lower()
        self.head_weight_decay = float(head_weight_decay)
        self._head_optimizer: Optional[torch.optim.Optimizer] = None
        self._head_param_ids: Optional[Tuple[int, ...]] = None

        # --- local learning
        self.local_lr = float(local_lr)
        self.alpha_between = float(alpha_between)
        self.beta_within = float(beta_within)
        self.mean_momentum = float(mean_momentum)
        self.sanger_diag_eps = float(sanger_diag_eps)

        # one-time freeze / param index (for heads)
        self._param_by_id: Optional[Dict[int, nn.Parameter]] = None
        self._frozen_once: bool = True
        self._trainable_ids: set[int] = set()

        # running means per block name
        self._mu_global_by_name: Dict[str, torch.Tensor] = {}
        self._mu_class_by_name: Dict[str, torch.Tensor] = {}
        self._num_classes_by_name: Dict[str, int] = {}

    # ============================================================
    # Heads optimizer (heads-only)
    # ============================================================

    def _collect_head_params(self, blocks) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen = set()
        for b in blocks:
            if not getattr(b, "is_output", False):
                continue
            for p in b.module.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def _ensure_head_optimizer(self, head_params: Sequence[nn.Parameter]) -> None:
        if self.head_optim == "none" or self.head_lr <= 0 or not head_params:
            self._head_optimizer = None
            self._head_param_ids = None
            return

        param_ids = tuple(sorted(id(p) for p in head_params))
        if self._head_optimizer is not None and self._head_param_ids == param_ids:
            return

        self._head_param_ids = param_ids

        if self.head_optim == "sgd":
            self._head_optimizer = torch.optim.SGD(
                head_params, lr=self.head_lr, weight_decay=self.head_weight_decay
            )
        elif self.head_optim == "ano":
            from ano_optimizer import Ano
            self._head_optimizer = Ano(
                head_params, lr=self.head_lr, weight_decay=self.head_weight_decay
            )
        else:
            self._head_optimizer = torch.optim.AdamW(
                head_params, lr=self.head_lr, weight_decay=self.head_weight_decay
            )

    # ============================================================
    # One-time freeze for heads
    # ============================================================

    def _ensure_param_index(self, model: nn.Module) -> None:
        if self._param_by_id is None:
            self._param_by_id = {id(p): p for p in model.parameters()}
            self._frozen_once = False
            self._trainable_ids = set()

    def _freeze_non_head_params_once(self, model: nn.Module, head_params: Sequence[nn.Parameter]) -> None:
        self._ensure_param_index(model)
        new_ids = {id(p) for p in head_params}

        if not self._frozen_once:
            for p in self._param_by_id.values():
                p.requires_grad = False
            for pid in new_ids:
                p = self._param_by_id.get(pid, None) if self._param_by_id is not None else None
                if p is not None:
                    p.requires_grad = True
            self._trainable_ids = set(new_ids)
            self._frozen_once = True
            return

        if new_ids != self._trainable_ids:
            for pid in (self._trainable_ids - new_ids):
                p = self._param_by_id.get(pid, None) if self._param_by_id is not None else None
                if p is not None:
                    p.requires_grad = False
            for pid in (new_ids - self._trainable_ids):
                p = self._param_by_id.get(pid, None) if self._param_by_id is not None else None
                if p is not None:
                    p.requires_grad = True
            self._trainable_ids = set(new_ids)

    # ============================================================
    # Loss / stats helper (heads)
    # ============================================================

    def _loss_from_outputs(self, task, out: Any, y: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        stats: Dict[str, Any] = {}

        if hasattr(out, "loss") and out.loss is not None and torch.is_tensor(out.loss):
            loss = out.loss
            if hasattr(out, "logits") and y is not None and torch.is_tensor(y):
                stats["acc"] = maybe_accuracy_from_logits(out.logits, y)
                stats.update(metric_payload_from_outputs(out.logits, y))
            return loss, stats

        logits = out.logits if hasattr(out, "logits") else out
        res = task.loss(logits, y)

        if torch.is_tensor(res):
            loss = res
            if torch.is_tensor(logits) and y is not None and torch.is_tensor(y):
                stats["acc"] = maybe_accuracy_from_logits(logits, y)
                stats.update(metric_payload_from_outputs(logits, y))
            return loss, stats

        if isinstance(res, tuple) and len(res) == 2 and torch.is_tensor(res[0]) and isinstance(res[1], Mapping):
            loss = res[0]
            stats.update({k: float(v) for k, v in res[1].items() if isinstance(v, (int, float))})
            if torch.is_tensor(logits) and y is not None and torch.is_tensor(y):
                stats.update(metric_payload_from_outputs(logits, y))
            return loss, stats

        if isinstance(res, Mapping) and "loss" in res:
            loss = res["loss"]
            if not torch.is_tensor(loss):
                loss = torch.tensor(float(loss), device=logits.device if torch.is_tensor(logits) else None)
            stats.update({k: float(v) for k, v in res.items() if k != "loss" and isinstance(v, (int, float))})
            if torch.is_tensor(logits) and y is not None and torch.is_tensor(y):
                stats.update(metric_payload_from_outputs(logits, y))
            return loss, stats

        raise TypeError(f"Unsupported loss return type from task.loss: {type(res)}")

    # ============================================================
    # Running means (global + per-class)
    # ============================================================

    def _ensure_means(self, name: str, xvec: torch.Tensor, labels: torch.Tensor) -> None:
        device = xvec.device
        D = xvec.size(1)
        K = int(labels.max().item()) + 1 if labels.numel() > 0 else 0

        if name not in self._mu_global_by_name:
            self._mu_global_by_name[name] = torch.zeros(D, device=device, dtype=xvec.dtype)

        if name not in self._mu_class_by_name or self._num_classes_by_name.get(name, 0) < K:
            old = self._mu_class_by_name.get(name, None)
            oldK = old.size(0) if old is not None else 0
            newK = max(K, oldK, 1)
            mu_class = torch.zeros(newK, D, device=device, dtype=xvec.dtype)
            if old is not None:
                mu_class[:oldK] = old.to(device=device, dtype=xvec.dtype)
            self._mu_class_by_name[name] = mu_class
            self._num_classes_by_name[name] = newK

        self._mu_global_by_name[name] = self._mu_global_by_name[name].to(device=device, dtype=xvec.dtype)
        self._mu_class_by_name[name] = self._mu_class_by_name[name].to(device=device, dtype=xvec.dtype)

    @torch.no_grad()
    def _update_means_ema(self, name: str, xvec: torch.Tensor, labels: torch.Tensor) -> None:
        m = self.mean_momentum
        if xvec.numel() == 0:
            return

        mu_g = self._mu_global_by_name[name]
        mu_g.mul_(1.0 - m).add_(xvec.mean(dim=0), alpha=m)

        mu_c = self._mu_class_by_name[name]
        K = mu_c.size(0)
        for k in range(K):
            mask = (labels == k)
            if mask.any():
                mu_c[k].mul_(1.0 - m).add_(xvec[mask].mean(dim=0), alpha=m)

    # ============================================================
    # Sanger/GHA helper
    # ============================================================

    def _sanger_update(self, Y: torch.Tensor, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Batch Sanger/GHA:
          ΔW = (Y^T X)/B - tril((Y^T Y)/B) W

        Shapes:
          X: [B, Din]
          Y: [B, Dout]
          W: [Dout, Din]
        """
        B = X.size(0)
        # correlations
        yx = (Y.t() @ X) / float(B + 0.0)           # [Dout, Din]
        yy = (Y.t() @ Y) / float(B + 0.0)           # [Dout, Dout]
        tril = torch.tril(yy)
        if self.sanger_diag_eps != 0.0:
            tril = tril + self.sanger_diag_eps * torch.eye(tril.size(0), device=tril.device, dtype=tril.dtype)
        return yx - (tril @ W)

    # ============================================================
    # Local supervised Sanger updates (no backward)
    # ============================================================

    @torch.no_grad()
    def _local_update_linear(self, layer: nn.Linear, name: str, xin: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        if xin.dim() != 2:
            return {}

        W = layer.weight.data  # [Dout, Din]
        B, Din = xin.shape
        Dout = W.size(0)

        # means
        self._ensure_means(name, xin, labels)
        self._update_means_ema(name, xin, labels)

        mu_g = self._mu_global_by_name[name]   # [Din]
        mu_c = self._mu_class_by_name[name]    # [K, Din]
        mu_y = mu_c[labels]                    # [B, Din]

        # signals
        r = xin - mu_y                         # within residuals
        b = mu_y - mu_g                        # between offsets

        # outputs
        yb = b @ W.t()                         # [B, Dout]
        yr = r @ W.t()                         # [B, Dout]

        # Sanger updates
        dW_between = self._sanger_update(yb, b, W)  # [Dout, Din]
        dW_within  = self._sanger_update(yr, r, W)  # [Dout, Din]

        dW = self.local_lr * (self.alpha_between * dW_between - self.beta_within * dW_within)
        W.add_(dW)

        # small bias decay (optional)
        if layer.bias is not None:
            layer.bias.data.mul_(1.0 - 0.001)

        # stats
        return {
            "sanger_yb2": float((yb * yb).mean().item()),
            "sanger_yr2": float((yr * yr).mean().item()),
            "sanger_norm": float(W.norm(dim=1).mean().item()),
        }

    @torch.no_grad()
    def _local_update_conv2d(self, conv: nn.Conv2d, name: str, xin: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        if xin.dim() != 4:
            return {}

        W = conv.weight.data  # [Cout, Cin, kH, kW]
        Cout, Cin, kH, kW = W.shape

        # pooled input
        x_eff = xin.mean(dim=(2, 3))  # [B, Cin]

        # means
        self._ensure_means(name, x_eff, labels)
        self._update_means_ema(name, x_eff, labels)

        mu_g = self._mu_global_by_name[name]   # [Cin]
        mu_c = self._mu_class_by_name[name]    # [K, Cin]
        mu_y = mu_c[labels]                    # [B, Cin]

        r = x_eff - mu_y
        b = mu_y - mu_g

        # averaged kernel
        Wbar = W.mean(dim=(2, 3))              # [Cout, Cin]

        yb = b @ Wbar.t()                      # [B, Cout]
        yr = r @ Wbar.t()                      # [B, Cout]

        dWb = self._sanger_update(yb, b, Wbar)  # [Cout, Cin]
        dWr = self._sanger_update(yr, r, Wbar)  # [Cout, Cin]

        dWbar = self.local_lr * (self.alpha_between * dWb - self.beta_within * dWr)

        # broadcast to full kernel
        W.add_(dWbar.view(Cout, Cin, 1, 1) / float(kH * kW))

        return {
            "sanger_yb2": float((yb * yb).mean().item()),
            "sanger_yr2": float((yr * yr).mean().item()),
            "sanger_norm": float(Wbar.norm(dim=1).mean().item()),
        }

    # ============================================================
    # main
    # ============================================================

    def train_step(self, model, task, batch, device, state=None) -> Dict[str, float]:
        model.train()

        if not hasattr(model, "get_blocks"):
            raise RuntimeError("KP3 requires models to expose get_blocks() (no fallback).")

        blocks = model.get_blocks()
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise RuntimeError("KP3 expects model.get_blocks() to return a non-empty list.")

        # move batch
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y = b.get("labels", None)
            x_for_forward = None
        else:
            x, y = batch
            x = to_device(x, device)
            y = to_device(y, device)
            x_for_forward = x

        # head optimizer
        head_params = self._collect_head_params(blocks)
        self._ensure_head_optimizer(head_params)
        self._freeze_non_head_params_once(model, head_params)

        stats: Dict[str, float] = {}
        loss_t: Optional[torch.Tensor] = None

        can_do_head_bp = (self._head_optimizer is not None) and (y is not None) and torch.is_tensor(y)

        if can_do_head_bp:
            if isinstance(batch, Mapping):
                out, cache = model(return_cache=True, **b)
            else:
                out, cache = model(x_for_forward, return_cache=True)

            loss, stats = self._loss_from_outputs(task, out, y)
            self._head_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._head_optimizer.step()
            loss_t = loss.detach()
        else:
            with torch.inference_mode():
                if isinstance(batch, Mapping):
                    out, cache = model(return_cache=True, **b)
                else:
                    out, cache = model(x_for_forward, return_cache=True)

            if y is not None and torch.is_tensor(y):
                loss_t, stats = self._loss_from_outputs(task, out, y)
            else:
                loss_t = None
                stats = {}

        if not isinstance(cache, Mapping) or "block_inputs" not in cache:
            raise RuntimeError("KP3 expects cache['block_inputs'] from model(..., return_cache=True).")

        block_inputs: Mapping[str, Any] = cache["block_inputs"]

        # local updates
        local_stats_accum: Dict[str, float] = {}
        local_updates = 0

        for bspec in blocks:
            if getattr(bspec, "is_output", False):
                continue

            if y is None or (not torch.is_tensor(y)):
                continue

            name = getattr(bspec, "name", None)
            mod = getattr(bspec, "module", None)
            if name is None or mod is None:
                continue

            xin = block_inputs.get(name, None)
            if not torch.is_tensor(xin):
                continue

            if isinstance(mod, nn.Linear):
                if xin.dim() != 2:
                    continue
                st = self._local_update_linear(mod, name, xin.detach(), y.detach())
            elif isinstance(mod, nn.Conv2d):
                if xin.dim() != 4:
                    continue
                st = self._local_update_conv2d(mod, name, xin.detach(), y.detach())
            else:
                st = {}

            if st:
                for k, v in st.items():
                    local_stats_accum[k] = local_stats_accum.get(k, 0.0) + float(v)
                local_updates += 1

        out_stats = dict(stats)
        out_stats["loss"] = float(loss_t.item()) if loss_t is not None else 0.0
        out_stats["local_updates"] = float(local_updates)

        if local_updates > 0:
            for k, v in local_stats_accum.items():
                out_stats[k] = float(v / local_updates)
        else:
            out_stats["sanger_yb2"] = 0.0
            out_stats["sanger_yr2"] = 0.0
            out_stats["sanger_norm"] = 0.0

        self.global_step += 1
        return out_stats


@register_update_rule("kp3")
def build_kp3(ctx: UpdateRuleContext):
    return KP3(local_lr=ctx.args.lr, head_lr=ctx.args.lr, head_weight_decay=ctx.args.weight_decay)
