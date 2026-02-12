# lab/algorithms/update_rules/softhebb_supcon.py
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


class KP2(UpdateRule):
    """
    SoftHebb backbone + local Supervised Contrastive Learning (SupCon) via *backprop locale*,
    but applied as an additive dW term while keeping SoftHebb's per-unit scaling (eta_k).

    Policy:
      - Output blocks (bspec.is_output=True): updated with standard backprop (heads-only optimizer).
      - Intermediate blocks:
          * Compute SoftHebb dW (no_grad, same as your SoftHebb).
          * Compute SupCon loss on a representation from the same block, get grads via autograd.grad
            (local backprop), then add dW_sup = -grad_W (descent direction).
          * Apply final update with SoftHebb eta_k scaling:
                W += eta_k[:, None] * (dW_soft + sup_strength * dW_sup)
    """

    def __init__(
        self,
        # -------------------------
        # SoftHebb core params
        # -------------------------
        learning_rate: float = 0.05,
        tau: float = 1.0,
        q: float = 0.5,
        anti_hebb: bool = True,
        # -------------------------
        # heads (backprop)
        # -------------------------
        head_lr: float = 3e-4,
        head_optim: str = "ano",  # "adamw" | "sgd" | "ano"
        head_weight_decay: float = 0.0,
        # -------------------------
        # depth LR scheduler (SoftHebb local LR)
        # -------------------------
        depth_lr_gamma: float = 0.5,
        depth_lr_min_factor: float = 0.01,
        depth_lr_max_factor: float = 100.0,
        # -------------------------
        # SupCon (local)
        # -------------------------
        supcon_tau: float = 0.5,
        sup_max: float = 0.1,
        sup_warmup_steps: int = 2_000,
        sup_apply_to_bias: bool = True,
        # -------------------------
        eps: float = 1e-8,
    ):
        super().__init__()
        # SoftHebb
        self.lr = float(learning_rate)
        self.tau = float(tau)
        self.q = float(q)
        self.anti_hebb = bool(anti_hebb)
        self.eps = float(eps)

        self.depth_lr_gamma = float(depth_lr_gamma)
        self.depth_lr_min_factor = float(depth_lr_min_factor)
        self.depth_lr_max_factor = float(depth_lr_max_factor)

        # heads
        self.head_lr = float(head_lr)
        self.head_optim = str(head_optim).lower()
        self.head_weight_decay = float(head_weight_decay)
        self._head_optimizer: Optional[torch.optim.Optimizer] = None
        self._head_param_ids: Optional[Tuple[int, ...]] = None

        # supcon
        self.supcon_tau = float(supcon_tau)
        self.sup_max = float(sup_max)
        self.sup_warmup_steps = int(sup_warmup_steps)
        self.sup_apply_to_bias = bool(sup_apply_to_bias)

        # one-time freeze / param index
        self._param_by_id: Optional[Dict[int, nn.Parameter]] = None
        self._frozen_once: bool = True
        self._trainable_ids: set[int] = set()

    # ============================================================
    # Helpers
    # ============================================================

    def _depth_scaled_lr(self, depth: int) -> float:
        factor = self.depth_lr_gamma ** int(depth)
        factor = max(self.depth_lr_min_factor, min(self.depth_lr_max_factor, factor))
        return float(self.lr * factor)

    def _sup_strength(self) -> float:
        if self.sup_warmup_steps <= 0:
            return self.sup_max
        t = min(1.0, float(self.global_step) / float(self.sup_warmup_steps))
        return float(self.sup_max * t)

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

    # ============================================================
    # SupCon loss (same spirit as your scl.py)
    # ============================================================

    def supervised_contrastive_loss(
        self,
        z: torch.Tensor,      # [B, D]
        labels: torch.Tensor, # [B]
        tau: float,
        eps: float,
    ) -> torch.Tensor:
        B = z.size(0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / tau
        sim = sim - sim.max(dim=1, keepdim=True)[0]  # stability

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0.0)

        exp_sim = torch.exp(sim) * (1 - torch.eye(B, device=z.device))
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + eps)
        return -mean_log_prob_pos.mean()

    # ============================================================
    # Local SupCon grads via autograd.grad
    # ============================================================

    def _supcon_grads_linear(
        self,
        layer: nn.Linear,
        xin: torch.Tensor,     # [B, Din]
        labels: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns: (grad_W, grad_b, loss_supcon)
        grad_W has shape [K, Din]
        """
        # recompute u with grad so grads flow to W
        u = layer(xin)  # [B, K]
        z = F.normalize(u, dim=1)
        loss = self.supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        params = [layer.weight]
        if layer.bias is not None:
            params.append(layer.bias)

        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)
        grad_w = grads[0]
        grad_b = grads[1] if (layer.bias is not None) else None
        return grad_w, grad_b, loss

    def _supcon_grads_conv2d(
        self,
        conv: nn.Conv2d,
        xin: torch.Tensor,     # [B, Cin, H, W]
        labels: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        SupCon on GAP(u) where u = conv(xin). Returns grads on conv weight/bias.
        grad_W has shape like conv.weight.
        """
        u = conv(xin)  # [B, Cout, H', W']
        v = u.mean(dim=(2, 3))  # [B, Cout]
        z = F.normalize(v, dim=1)
        loss = self.supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        params = [conv.weight]
        if conv.bias is not None:
            params.append(conv.bias)

        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)
        grad_w = grads[0]
        grad_b = grads[1] if (conv.bias is not None) else None
        return grad_w, grad_b, loss

    # ============================================================
    # SoftHebb + additive SupCon dW (Linear / Conv)
    # ============================================================

    def _local_update_linear_softhebb_plus_supcon(
        self,
        layer: nn.Linear,
        xin: torch.Tensor,                 # [B, D]
        u_cache: torch.Tensor,             # [B, K] (from cache, no grad ok)
        labels: Optional[torch.Tensor],
        base_lr: float,
        sup_strength: float,
    ) -> float:
        """
        Applies:
          dW = dW_softhebb + sup_strength * (-grad_supcon)
          W += eta_k * dW
        Returns supcon loss value for logging (0 if skipped).
        """
        B = xin.size(0)
        W = layer.weight  # [K, D]

        # ---- SoftHebb core (IDENTICAL to your implementation)
        y = self._soft_wta(u_cache)  # [B, K]
        if self.anti_hebb:
            y.mul_(self._winner_sign(y))
        y_eff = y

        term1 = (y_eff.T @ xin) / float(B)  # [K, D]
        a = (y_eff * u_cache).mean(dim=0)   # [K]
        dW = term1 - a[:, None] * W         # [K, D]

        db = None
        if layer.bias is not None:
            mean_y = y_eff.mean(dim=0)
            db = mean_y - a * layer.bias  # [K]

        # ---- SupCon local backprop -> grads
        sup_loss_val = 0.0
        if labels is not None and sup_strength > 0.0 and B >= 2:
            # enable grad locally, recompute forward with grad
            with torch.enable_grad():
                # Important: xin must be detached (we only want grads wrt layer params)
                xin_g = xin.detach()
                # Make sure params are grad-enabled just for autograd.grad
                # (they can have requires_grad=False globally due to freezing)
                w_req = layer.weight.requires_grad
                b_req = layer.bias.requires_grad if layer.bias is not None else None
                layer.weight.requires_grad_(True)
                if layer.bias is not None:
                    layer.bias.requires_grad_(True)

                grad_w, grad_b, sup_loss = self._supcon_grads_linear(layer, xin_g, labels)
                sup_loss_val = float(sup_loss.detach().item())

                # restore requires_grad flags
                layer.weight.requires_grad_(w_req)
                if layer.bias is not None and b_req is not None:
                    layer.bias.requires_grad_(b_req)

            print(f"grad_w supcon norm: {grad_w.norm().item():.4f}")
            print(f"grad_b supcon norm: {grad_b.norm().item():.4f}" if grad_b is not None else "no bias")
            #print(f"softhebb dW norm: {dW.norm().item():.4f}")
            print(f"sup_strength: {sup_strength:.4f}")
            print(f"supcon dW norm: {(sup_strength * grad_w).norm().item():.4f}")
            # descent direction is -grad
            dW = 0
            #print(f"combined dW norm: {dW.norm().item():.4f}")
            if layer.bias is not None and self.sup_apply_to_bias and (grad_b is not None) and (db is not None):
                db = 0

        # ---- Apply per-unit eta scaling (SoftHebb style)
        #eta_k = self._eta_per_unit_from_weight(W, base_lr=base_lr)  # [K]
        with torch.no_grad():
            layer.weight.add_(base_lr* dW)
            if layer.bias is not None and db is not None:
                layer.bias.add_(base_lr * db)

        return sup_loss_val

    def _local_update_conv2d_softhebb_plus_supcon(
        self,
        conv: nn.Conv2d,
        xin: torch.Tensor,                 # [B, Cin, H, W]
        u_cache: torch.Tensor,             # [B, Cout, H', W'] from cache
        labels: Optional[torch.Tensor],
        base_lr: float,
        sup_strength: float,
    ) -> float:
        """
        Applies SoftHebb update (same as yours) on conv weights + additive SupCon gradient.
        SupCon is computed on GAP(conv(xin)) with local backprop.
        """
        B = xin.size(0)
        Cout = u_cache.size(1)

        # ---- SoftHebb core (IDENTICAL to your implementation)
        patches = F.unfold(
            xin,
            kernel_size=conv.kernel_size,
            dilation=conv.dilation,
            padding=conv.padding,
            stride=conv.stride,
        )  # [B, D, L]
        L = patches.size(2)

        u_ = u_cache.reshape(B, Cout, L)  # [B, Cout, L]
        y = self._soft_wta(u_)            # [B, Cout, L]
        y_eff = y * self._winner_sign(y) if self.anti_hebb else y

        W_flat = conv.weight.view(Cout, -1)  # [Cout, D]

        term1 = torch.einsum("bkl,bdl->kd", y_eff, patches) / float(B * L)
        a = (y_eff * u_).mean(dim=(0, 2))  # [Cout]
        dW_flat = term1 - a[:, None] * W_flat

        db = None
        if conv.bias is not None:
            mean_y = y_eff.mean(dim=(0, 2))
            db = mean_y - a * conv.bias  # [Cout]

        # ---- SupCon local backprop -> grads
        sup_loss_val = 0.0
        if labels is not None and sup_strength > 0.0 and B >= 2:
            with torch.enable_grad():
                xin_g = xin.detach()

                w_req = conv.weight.requires_grad
                b_req = conv.bias.requires_grad if conv.bias is not None else None
                conv.weight.requires_grad_(True)
                if conv.bias is not None:
                    conv.bias.requires_grad_(True)

                grad_w, grad_b, sup_loss = self._supcon_grads_conv2d(conv, xin_g, labels)
                sup_loss_val = float(sup_loss.detach().item())

                conv.weight.requires_grad_(w_req)
                if conv.bias is not None and b_req is not None:
                    conv.bias.requires_grad_(b_req)

            # Map grad_w into flat shape and subtract (descent)
            grad_w_flat = grad_w.detach().view(Cout, -1)
            dW_flat = dW_flat + (sup_strength * (-grad_w_flat))

            if conv.bias is not None and self.sup_apply_to_bias and (grad_b is not None) and (db is not None):
                db = db + (sup_strength * (-grad_b.detach()))

        # ---- Apply per-unit eta scaling (SoftHebb style)
        eta_k = self._eta_per_unit_from_weight(W_flat, base_lr=base_lr)  # [Cout]
        with torch.no_grad():
            W_flat.add_(eta_k[:, None] * dW_flat)
            conv.weight.copy_(W_flat.view_as(conv.weight))
            if conv.bias is not None and db is not None:
                conv.bias.add_(eta_k * db)

        return sup_loss_val

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
            raise RuntimeError("SoftHebbSupCon requires models to expose get_blocks() (no fallback).")

        blocks = model.get_blocks()
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise RuntimeError("SoftHebbSupCon expects model.get_blocks() to return a non-empty list.")

        # --- Move batch once
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y = b.get("labels", None)
            x_for_forward = None
        else:
            x, y = batch
            x = to_device(x, device)
            y = to_device(y, device)
            x_for_forward = x

        # --- Heads-only optimizer + one-time freeze
        head_params = self._collect_head_params(blocks)
        self._ensure_head_optimizer(head_params)
        self._freeze_non_head_params_once(model, head_params)

        stats: Dict[str, float] = {}
        loss_t: Optional[torch.Tensor] = None

        can_do_bp = (self._head_optimizer is not None) and (y is not None)

        # --- Forward for heads + cache
        if can_do_bp:
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

            if y is not None:
                loss_t, stats = self._loss_from_outputs(task, out, y)
            else:
                loss_t = None
                stats = {}

        if not isinstance(cache, Mapping) or "block_inputs" not in cache:
            raise RuntimeError("SoftHebbSupCon expects cache['block_inputs'] from model(..., return_cache=True).")
        if "block_outputs" not in cache:
            raise RuntimeError("SoftHebbSupCon expects cache['block_outputs'] from model(..., return_cache=True).")

        block_inputs: Mapping[str, Any] = cache["block_inputs"]
        block_outputs: Mapping[str, Any] = cache["block_outputs"]

        # --- Local updates: SoftHebb + SupCon grads
        hebb_lrs: List[float] = []
        supcon_losses: List[float] = []
        depth_idx = 0
        sup_strength = self._sup_strength()

        # Only do SupCon if we have labels
        labels = y if (y is not None and torch.is_tensor(y)) else None

        for bspec in blocks:
            if getattr(bspec, "is_output", False):
                continue

            name = getattr(bspec, "name", None)
            mod = getattr(bspec, "module", None)
            if name is None or mod is None:
                continue

            xin = block_inputs.get(name, None)
            u = block_outputs.get(name, None)
            if not torch.is_tensor(xin) or not torch.is_tensor(u):
                continue

            lr_here = self._depth_scaled_lr(depth_idx)

            # Linear blocks
            if isinstance(mod, nn.Linear):
                if xin.dim() == 2 and u.dim() == 2:
                    sup_l = self._local_update_linear_softhebb_plus_supcon(
                        layer=mod,
                        xin=xin,
                        u_cache=u,
                        labels=labels,
                        base_lr=lr_here,
                        sup_strength=sup_strength,
                    )
                    hebb_lrs.append(lr_here)
                    if sup_l > 0.0:
                        supcon_losses.append(sup_l)
                    depth_idx += 1
                continue

            # ConvBlock blocks
            if isinstance(mod, ConvBlock):
                if xin.dim() == 4 and u.dim() == 4:
                    sup_l = self._local_update_conv2d_softhebb_plus_supcon(
                        conv=mod.conv,
                        xin=xin,
                        u_cache=u,
                        labels=labels,
                        base_lr=lr_here,
                        sup_strength=sup_strength,
                    )
                    hebb_lrs.append(lr_here)
                    if sup_l > 0.0:
                        supcon_losses.append(sup_l)
                    depth_idx += 1
                continue

            # Optionally support raw nn.Conv2d too
            if isinstance(mod, nn.Conv2d):
                if xin.dim() == 4 and u.dim() == 4:
                    sup_l = self._local_update_conv2d_softhebb_plus_supcon(
                        conv=mod,
                        xin=xin,
                        u_cache=u,
                        labels=labels,
                        base_lr=lr_here,
                        sup_strength=sup_strength,
                    )
                    hebb_lrs.append(lr_here)
                    if sup_l > 0.0:
                        supcon_losses.append(sup_l)
                    depth_idx += 1
                continue

            # else ignore
            continue

        out_stats = dict(stats)
        out_stats["loss"] = float(loss_t.item()) if loss_t is not None else 0.0
        out_stats["sup_strength"] = float(sup_strength)
        out_stats["supcon_loss"] = float(sum(supcon_losses) / len(supcon_losses)) if supcon_losses else 0.0
        out_stats["hebb_lr_min"] = float(min(hebb_lrs)) if hebb_lrs else 0.0
        out_stats["hebb_lr_max"] = float(max(hebb_lrs)) if hebb_lrs else 0.0

        self.global_step += 1
        return out_stats
