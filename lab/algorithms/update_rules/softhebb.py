# lab/algorithms/update_rules/softhebb.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule
from ...core.batch import to_device
from ...core.steps import maybe_accuracy_from_logits
from ...models.convnet import ConvBlock


class SoftHebb(UpdateRule):
    """
    SoftHebb (block-based, minimal, no RL logic):

    Policy:
      - if block.is_output: head updated with BACKPROP (heads-only optimizer)
      - else: SoftHebb update if module is nn.Linear or ConvBlock (no_grad)

    Adds depth-adaptive local LR:
      lr(depth) = base_lr * (depth_lr_gamma ** depth), clamped.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        tau: float = 1.0,
        q: float = 0.5,
        anti_hebb: bool = True,
        # heads (backprop)
        head_lr: float = 3e-4,
        head_optim: str = "ano",  # "adamw" | "sgd" | "ano"
        head_weight_decay: float = 0.0,
        # depth LR scheduler (NEW)
        depth_lr_gamma: float = 0.5,
        depth_lr_min_factor: float = 0.01,
        depth_lr_max_factor: float = 100.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lr = float(learning_rate)  # base local lr
        self.tau = float(tau)
        self.q = float(q)
        self.anti_hebb = bool(anti_hebb)
        self.eps = float(eps)

        # depth scheduling params (NEW)
        self.depth_lr_gamma = float(depth_lr_gamma)
        self.depth_lr_min_factor = float(depth_lr_min_factor)
        self.depth_lr_max_factor = float(depth_lr_max_factor)

        # heads
        self.head_lr = float(head_lr)
        self.head_optim = str(head_optim).lower()
        self.head_weight_decay = float(head_weight_decay)

        self._head_optimizer: Optional[torch.optim.Optimizer] = None
        self._head_param_ids: Optional[Tuple[int, ...]] = None

        # one-time freeze / param index
        self._param_by_id: Optional[Dict[int, nn.Parameter]] = None
        self._frozen_once: bool = True
        self._trainable_ids: set[int] = set()

    # ============================================================
    # Depth LR scheduler helpers (NEW)
    # ============================================================

    def _depth_scaled_lr(self, depth: int) -> float:
        """
        lr(depth) = base_lr * gamma^depth, clamped to [min_factor, max_factor] * base_lr.
        """
        factor = self.depth_lr_gamma ** int(depth)
        factor = max(self.depth_lr_min_factor, min(self.depth_lr_max_factor, factor))
        return float(self.lr * factor)

    # ============================================================
    # SoftHebb core
    # ============================================================

    def _eta_per_unit_from_weight(self, w_2d: torch.Tensor, base_lr: float) -> torch.Tensor:
        """
        Per-unit learning rate:
          eta_k = base_lr * (max(||w_k|| - 1, 0) + eps)^q
        """
        r = torch.norm(w_2d, dim=1)  # [K]
        base = torch.clamp(r - 1.0, min=0.0)
        if self.q == 0.0:
            return torch.full_like(r, base_lr)
        return base_lr * (base + self.eps).pow(self.q)

    def _soft_wta(self, u: torch.Tensor) -> torch.Tensor:
        return F.softmax(u / self.tau, dim=1)

    def _winner_sign(self, y: torch.Tensor) -> torch.Tensor:
        S = y.new_full(y.shape, -1.0)
        idx = y.argmax(dim=1, keepdim=True)
        S.scatter_(1, idx, 1.0)
        return S

    @torch.no_grad()
    def _update_linear_softhebb(self, layer: nn.Linear, x: torch.Tensor, u: torch.Tensor, base_lr: float):
        # x: [B,D], u: [B,K]
        B = x.size(0)
        W = layer.weight  # [K,D]

        y = self._soft_wta(u)  # [B,K]
        if self.anti_hebb:
            y.mul_(self._winner_sign(y))
        y_eff = y

        term1 = (y_eff.T @ x) / float(B)  # [K,D]
        a = (y_eff * u).mean(dim=0)  # [K]
        dW = term1 - a[:, None] * W  # [K,D]

        eta_k = self._eta_per_unit_from_weight(W, base_lr=base_lr)
        layer.weight.add_(eta_k[:, None] * dW)

        if layer.bias is not None:
            mean_y = y_eff.mean(dim=0)
            db = mean_y - a * layer.bias
            layer.bias.add_(eta_k * db)

    @torch.no_grad()
    def _update_conv2d_softhebb(self, layer: nn.Conv2d, x: torch.Tensor, u: torch.Tensor, base_lr: float):
        # x: [B,Cin,H,W], u: [B,Cout,H',W']
        B = x.size(0)
        Cout = u.size(1)

        patches = F.unfold(
            x,
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )  # [B, D, L]

        L = patches.size(2)
        u_ = u.reshape(B, Cout, L)  # [B,Cout,L]

        y = self._soft_wta(u_)
        y_eff = y * self._winner_sign(y) if self.anti_hebb else y

        W_flat = layer.weight.view(Cout, -1)  # [Cout,D]

        term1 = torch.einsum("bkl,bdl->kd", y_eff, patches) / float(B * L)
        a = (y_eff * u_).mean(dim=(0, 2))
        dW_flat = term1 - a[:, None] * W_flat

        eta_k = self._eta_per_unit_from_weight(W_flat, base_lr=base_lr)
        W_flat.add_(eta_k[:, None] * dW_flat)
        layer.weight.copy_(W_flat.view_as(layer.weight))

        if layer.bias is not None:
            mean_y = y_eff.mean(dim=(0, 2))
            db = mean_y - a * layer.bias
            layer.bias.add_(eta_k * db)

    # ============================================================
    # Head optimizer (heads-only)
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

    def _ensure_head_optimizer(self, head_params: Sequence[nn.Parameter]):
        if not head_params:
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
    # One-time freeze (avoid per-step requires_grad toggling)
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
    # Loss / stats helper
    # ============================================================

    def _loss_from_outputs(self, task, out: Any, y: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats: Dict[str, float] = {}

        if hasattr(out, "loss") and out.loss is not None and torch.is_tensor(out.loss):
            loss = out.loss
            if hasattr(out, "logits") and y is not None and torch.is_tensor(y):
                stats["acc"] = maybe_accuracy_from_logits(out.logits, y)
            return loss, stats

        logits = out.logits if hasattr(out, "logits") else out
        res = task.loss(logits, y)

        if torch.is_tensor(res):
            loss = res
            if torch.is_tensor(logits) and y is not None and torch.is_tensor(y):
                stats["acc"] = maybe_accuracy_from_logits(logits, y)
            return loss, stats

        if isinstance(res, tuple) and len(res) == 2 and torch.is_tensor(res[0]) and isinstance(res[1], Mapping):
            loss = res[0]
            stats.update({k: float(v) for k, v in res[1].items() if isinstance(v, (int, float))})
            return loss, stats

        if isinstance(res, Mapping) and "loss" in res:
            loss = res["loss"]
            if not torch.is_tensor(loss):
                loss = torch.tensor(float(loss), device=logits.device if torch.is_tensor(logits) else None)
            stats.update({k: float(v) for k, v in res.items() if k != "loss" and isinstance(v, (int, float))})
            return loss, stats

        raise TypeError(f"Unsupported loss return type from task.loss: {type(res)}")

    # ============================================================
    # main
    # ============================================================

    def train_step(self, model, task, batch, device) -> Dict[str, float]:
        model.train()

        if not hasattr(model, "get_blocks"):
            raise RuntimeError("SoftHebb requires models to expose get_blocks() (no fallback).")

        blocks = model.get_blocks()
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise RuntimeError("SoftHebb expects model.get_blocks() to return a non-empty list.")

        # --- Move batch once
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y = b.get("labels", None)
        else:
            x, y = batch
            x = to_device(x, device)
            y = to_device(y, device)

        # --- Heads-only optimizer + one-time freeze
        head_params = self._collect_head_params(blocks)
        self._ensure_head_optimizer(head_params)
        self._freeze_non_head_params_once(model, head_params)

        stats: Dict[str, float] = {}
        loss_t: Optional[torch.Tensor] = None

        can_do_bp = (self._head_optimizer is not None) and (y is not None)

        if can_do_bp:
            if isinstance(batch, Mapping):
                out, cache = model(return_cache=True, **b)
            else:
                out, cache = model(x, return_cache=True)

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
                    out, cache = model(x, return_cache=True)

            if y is not None:
                loss_t, stats = self._loss_from_outputs(task, out, y)
            else:
                loss_t = None
                stats = {}

        if not isinstance(cache, Mapping) or "block_inputs" not in cache:
            raise RuntimeError("SoftHebb expects cache['block_inputs'] from model(..., return_cache=True).")

        block_inputs: Mapping[str, Any] = cache["block_inputs"]

        # --- SoftHebb updates with depth-adaptive LR (NEW)
        hebb_lrs: List[float] = []
        depth_idx = 0

        with torch.no_grad():
            for bspec in blocks:
                if getattr(bspec, "is_output", False):
                    continue

                name = getattr(bspec, "name", None)
                mod = getattr(bspec, "module", None)
                if name is None or mod is None:
                    continue

                xin = block_inputs.get(name, None)
                if not torch.is_tensor(xin):
                    continue

                if "block_outputs" not in cache or name not in cache["block_outputs"]:
                    continue
                u = cache["block_outputs"][name]
                if not torch.is_tensor(u):
                    continue

                lr_here = self._depth_scaled_lr(depth_idx)

                if isinstance(mod, nn.Linear):
                    if xin.dim() == 2 and u.dim() == 2:
                        self._update_linear_softhebb(mod, xin, u, base_lr=lr_here)
                        hebb_lrs.append(lr_here)
                        depth_idx += 1

                elif isinstance(mod, ConvBlock):
                    if xin.dim() == 4 and u.dim() == 4:
                        self._update_conv2d_softhebb(mod.conv, xin, u, base_lr=lr_here)
                        hebb_lrs.append(lr_here)
                        depth_idx += 1

                else:
                    continue

        out_stats = dict(stats)
        out_stats["loss"] = float(loss_t.item()) if loss_t is not None else 0.0
        # optional debug stats (NEW)
        out_stats["hebb_lr_min"] = float(min(hebb_lrs)) if hebb_lrs else 0.0
        out_stats["hebb_lr_max"] = float(max(hebb_lrs)) if hebb_lrs else 0.0

        self.global_step += 1
        return out_stats
