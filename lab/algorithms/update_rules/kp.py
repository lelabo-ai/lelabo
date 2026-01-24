# lab/algorithms/update_rules/kp.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule
from ...core.batch import to_device
from ...core.steps import maybe_accuracy_from_logits


class KP(UpdateRule):
    """
    KP: SoftHebb + 2-phase schedule (supervised, block-based only)

    Requirements:
      - model.get_blocks() -> list of BlockSpec-like objects with:
          name: str
          module: nn.Module
          is_output: bool
      - model.forward(..., return_cache=True) -> (out, cache)
      - cache["block_inputs"][block_name] exists for blocks you want to update

    Phases:
      A) pretrain: SoftHebb on TRUNK only (blocks where is_output=False),
         until stagnation criterion on sum_eta.
         Behavior: train_step() loops internally until switch triggers, then performs
         exactly one finetune BP step on the same batch and returns.

      B) finetune: one BP step per call (scope=head or all), optionally keep SoftHebb
         trunk updates (continue_softhebb_after_pretrain).
    """

    def __init__(
        self,
        # --- SoftHebb core ---
        learning_rate: float = 0.05,
        tau: float = 1.0,
        q: float = 0.5,
        anti_hebb: bool = True,
        eps: float = 1e-8,
        # --- trunk pretrain schedule ---
        trunk_pretrain: bool = True,
        trunk_pretrain_min_steps: int = 200,
        trunk_pretrain_patience: int = 50,
        trunk_pretrain_rel_tol: float = 1e-3,
        trunk_pretrain_ema_beta: float = 0.05,
        trunk_pretrain_max_steps: int = 10_000,
        # --- finetune (BP) ---
        bp_after_pretrain: bool = True,
        bp_scope: str = "head",   # "head" | "all"
        bp_lr: float = 1e-3,
        bp_optim: str = "adamw",  # "adamw" | "sgd"
        bp_weight_decay: float = 0.0,
        # keep SoftHebb trunk updates during BP phase
        continue_softhebb_after_pretrain: bool = False,
        # optional: print every N inner-pretrain steps
        verbose_every: int = 0,
    ):
        super().__init__()
        self.lr = float(learning_rate)
        self.tau = float(tau)
        self.q = float(q)
        self.anti_hebb = bool(anti_hebb)
        self.eps = float(eps)

        self.trunk_pretrain = bool(trunk_pretrain)
        self.trunk_pretrain_min_steps = int(trunk_pretrain_min_steps)
        self.trunk_pretrain_patience = int(trunk_pretrain_patience)
        self.trunk_pretrain_rel_tol = float(trunk_pretrain_rel_tol)
        self.trunk_pretrain_ema_beta = float(trunk_pretrain_ema_beta)
        self.trunk_pretrain_max_steps = int(trunk_pretrain_max_steps)

        self.bp_after_pretrain = bool(bp_after_pretrain)
        self.bp_scope = str(bp_scope).lower()
        if self.bp_scope not in ("head", "all"):
            raise ValueError("bp_scope must be 'head' or 'all'")
        self.bp_lr = float(bp_lr)
        self.bp_optim = str(bp_optim).lower()
        self.bp_weight_decay = float(bp_weight_decay)

        self.continue_softhebb_after_pretrain = bool(continue_softhebb_after_pretrain)
        self.verbose_every = int(verbose_every)

        # BP optimizer (lazy)
        self._bp_optimizer: Optional[torch.optim.Optimizer] = None
        self._bp_param_ids: Optional[Tuple[int, ...]] = None

        # phase state
        self._phase: str = "pretrain" if self.trunk_pretrain else "finetune"
        self._pretrain_steps: int = 0
        self._eta_ema: Optional[float] = None
        self._stall_steps: int = 0

    # ============================================================
    # SoftHebb core
    # ============================================================

    def _eta_per_unit_from_weight(self, w_2d: torch.Tensor) -> torch.Tensor:
        r = torch.norm(w_2d, dim=1)  # [K]
        base = torch.clamp(r - 1.0, min=0.0)
        if self.q == 0.0:
            return torch.full_like(r, self.lr)
        return self.lr * (base + self.eps).pow(self.q)

    def _soft_wta(self, u: torch.Tensor) -> torch.Tensor:
        return F.softmax(u / self.tau, dim=1)

    def _winner_sign(self, y: torch.Tensor) -> torch.Tensor:
        S = -torch.ones_like(y)
        if y.dim() == 2:
            idx = y.argmax(dim=1)
            S[torch.arange(y.size(0), device=y.device), idx] = 1.0
        elif y.dim() == 3:
            idx = y.argmax(dim=1)
            S.scatter_(1, idx.unsqueeze(1), 1.0)
        return S

    @torch.no_grad()
    def _update_linear_softhebb(self, layer: nn.Linear, x: torch.Tensor):
        # x: [B,D]
        u = layer(x)  # [B,K]
        if u.dim() != 2:
            return

        B = x.size(0)
        W = layer.weight  # [K,D]

        y = self._soft_wta(u)  # [B,K]
        y_eff = y * self._winner_sign(y) if self.anti_hebb else y

        term1 = (y_eff.T @ x) / float(B)          # [K,D]
        a = (y_eff * u).mean(dim=0)               # [K]
        dW = term1 - a[:, None] * W               # [K,D]

        eta_k = self._eta_per_unit_from_weight(W)
        layer.weight.add_(eta_k[:, None] * dW)

        if layer.bias is not None:
            mean_y = y_eff.mean(dim=0)
            db = mean_y - a * layer.bias
            layer.bias.add_(eta_k * db)

    @torch.no_grad()
    def _update_conv2d_softhebb(self, layer: nn.Conv2d, x: torch.Tensor):
        # x: [B,Cin,H,W]
        u = layer(x)  # [B,Cout,H',W']
        if u.dim() != 4:
            return

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

        eta_k = self._eta_per_unit_from_weight(W_flat)
        W_flat.add_(eta_k[:, None] * dW_flat)
        layer.weight.copy_(W_flat.view_as(layer.weight))

        if layer.bias is not None:
            mean_y = y_eff.mean(dim=(0, 2))
            db = mean_y - a * layer.bias
            layer.bias.add_(eta_k * db)

    # ============================================================
    # Stagnation criterion
    # ============================================================

    @torch.no_grad()
    def _sum_eta_trunk(self, trunk_blocks) -> float:
        total = 0.0
        for b in trunk_blocks:
            m = b.module
            if isinstance(m, nn.Linear):
                eta = self._eta_per_unit_from_weight(m.weight)
                total += float(eta.sum().item())
            elif isinstance(m, nn.Conv2d):
                W_flat = m.weight.view(m.weight.size(0), -1)
                eta = self._eta_per_unit_from_weight(W_flat)
                total += float(eta.sum().item())
        return total

    def _maybe_switch_phase(self, sum_eta: float) -> bool:
        if self._eta_ema is None:
            self._eta_ema = sum_eta
            self._stall_steps = 0
            return False

        beta = self.trunk_pretrain_ema_beta
        ema = (1.0 - beta) * self._eta_ema + beta * sum_eta
        rel = abs(sum_eta - ema) / (abs(ema) + self.eps)

        self._eta_ema = ema
        if rel < self.trunk_pretrain_rel_tol:
            self._stall_steps += 1
        else:
            self._stall_steps = 0

        if self._pretrain_steps < self.trunk_pretrain_min_steps:
            return False

        if self._stall_steps >= self.trunk_pretrain_patience:
            self._phase = "finetune"
            return True

        if self._pretrain_steps >= self.trunk_pretrain_max_steps:
            self._phase = "finetune"
            return True

        return False

    # ============================================================
    # BP optimizer (finetune)
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

    def _ensure_bp_optimizer(self, params: Sequence[nn.Parameter]):
        if not params:
            self._bp_optimizer = None
            self._bp_param_ids = None
            return

        param_ids = tuple(sorted(id(p) for p in params))
        if self._bp_optimizer is not None and self._bp_param_ids == param_ids:
            return

        self._bp_param_ids = param_ids
        if self.bp_optim == "sgd":
            self._bp_optimizer = torch.optim.SGD(params, lr=self.bp_lr, weight_decay=self.bp_weight_decay)
        elif self.bp_optim == 'ano':
            from ano_optimizer import Ano
            self._bp_optimizer = Ano(params, lr=self.bp_lr, weight_decay=self.bp_weight_decay)
        else:
            self._bp_optimizer = torch.optim.AdamW(params, lr=self.bp_lr, weight_decay=self.bp_weight_decay)

    def _set_requires_grad_only(self, model: nn.Module, allow_param_ids: set[int]) -> Dict[int, bool]:
        old: Dict[int, bool] = {}
        for p in model.parameters():
            old[id(p)] = p.requires_grad
            p.requires_grad = (id(p) in allow_param_ids)
        return old

    def _restore_requires_grad(self, model: nn.Module, old: Dict[int, bool]) -> None:
        for p in model.parameters():
            p.requires_grad = old.get(id(p), p.requires_grad)

    # ============================================================
    # Loss helper (supervised)
    # ============================================================

    def _loss_and_stats(self, task, out: Any, y: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats: Dict[str, float] = {}

        # HF outputs with .loss
        if hasattr(out, "loss") and out.loss is not None and torch.is_tensor(out.loss):
            loss = out.loss
            if hasattr(out, "logits") and y is not None and torch.is_tensor(y):
                stats["acc"] = maybe_accuracy_from_logits(out.logits, y)
            return loss, stats

        logits = out.logits if hasattr(out, "logits") else out
        loss = task.loss(logits, y)
        if torch.is_tensor(logits) and y is not None and torch.is_tensor(y):
            stats["acc"] = maybe_accuracy_from_logits(logits, y)
        return loss, stats

    # ============================================================
    # One SoftHebb trunk step
    # ============================================================

    @torch.no_grad()
    def _softhebb_trunk_step(self, blocks, block_inputs: Mapping[str, Any]) -> None:
        for b in blocks:
            if getattr(b, "is_output", False):
                continue
            name = getattr(b, "name", None)
            m = getattr(b, "module", None)
            if name is None or m is None:
                continue
            if name not in block_inputs:
                continue

            x = block_inputs[name]
            if not torch.is_tensor(x):
                continue

            if isinstance(m, nn.Linear):
                if x.dim() == 2:
                    self._update_linear_softhebb(m, x)
            elif isinstance(m, nn.Conv2d):
                if x.dim() == 4:
                    self._update_conv2d_softhebb(m, x)

    # ============================================================
    # Main
    # ============================================================

    def train_step(self, model, task, batch, device) -> Dict[str, float]:
        model.train()

        if not hasattr(model, "get_blocks"):
            raise RuntimeError("KP requires model.get_blocks() (no fallback).")
        blocks = model.get_blocks()
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise RuntimeError("KP expects model.get_blocks() to return a non-empty list.")

        # prepare batch
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y = b.get("labels", None)
        else:
            x, y = batch
            x = to_device(x, device)
            y = to_device(y, device)

        # define trunk blocks (only those SoftHebb can update)
        trunk_blocks = [
            bb for bb in blocks
            if (not getattr(bb, "is_output", False)) and isinstance(bb.module, (nn.Linear, nn.Conv2d))
        ]

        # ------------------------------------------------------------
        # PHASE A: internal loop until switch, then 1 finetune step
        # ------------------------------------------------------------
        if self._phase == "pretrain":
            inner = 0
            last_stats: Dict[str, float] = {"loss": 0.0}

            while self._phase == "pretrain":
                # forward no_grad with cache
                with torch.no_grad():
                    if isinstance(batch, Mapping):
                        out_ng, cache = model(return_cache=True, **b)
                    else:
                        out_ng, cache = model(x, return_cache=True)

                if not isinstance(cache, Mapping) or "block_inputs" not in cache:
                    raise RuntimeError("KP expects cache['block_inputs'] from model(..., return_cache=True).")
                block_inputs = cache["block_inputs"]

                # SoftHebb trunk update
                self._softhebb_trunk_step(blocks, block_inputs)

                # stagnation update
                self._pretrain_steps += 1
                sum_eta = self._sum_eta_trunk(trunk_blocks)
                just_switched = self._maybe_switch_phase(sum_eta)

                # log stats (loss only for monitoring)
                loss_t = None
                stats: Dict[str, float] = {}
                if y is not None:
                    loss_t, stats = self._loss_and_stats(task, out_ng, y)
                else:
                    loss_t = torch.tensor(0.0, device=device)

                stats_out = {
                    "loss": float(loss_t.item()),
                    "phase": 0.0,
                    "pretrain_steps": float(self._pretrain_steps),
                    "pretrain_inner_steps": float(inner + 1),
                    "sum_eta": float(sum_eta),
                    "eta_ema": float(self._eta_ema if self._eta_ema is not None else sum_eta),
                    "eta_stall": float(self._stall_steps),
                    "switched": 1.0 if just_switched else 0.0,
                }
                stats_out.update(stats)
                last_stats = stats_out

                inner += 1
                self.global_step += 1

                if self.verbose_every > 0 and (inner % self.verbose_every) == 0:
                    print(
                        f"[KP] phase=pretrain inner={inner} pretrain_steps={self._pretrain_steps} "
                        f"sum_eta={sum_eta:.4e} ema={stats_out['eta_ema']:.4e} stall={self._stall_steps}"
                    )

            # switched -> finetune one step on same batch (if enabled)
            if self._phase == "finetune" and self.bp_after_pretrain:
                fin = self._finetune_one_step(model, task, batch, device, blocks, y, b if isinstance(batch, Mapping) else None, x if not isinstance(batch, Mapping) else None)
                fin["pretrain_inner_steps"] = float(inner)
                return fin

            return last_stats

        # ------------------------------------------------------------
        # PHASE B: finetune one step per call
        # ------------------------------------------------------------
        if self._phase == "finetune" and self.bp_after_pretrain:
            return self._finetune_one_step(model, task, batch, device, blocks, y, b if isinstance(batch, Mapping) else None, x if not isinstance(batch, Mapping) else None)

        # If trunk_pretrain=False and bp_after_pretrain=False, just do trunk SoftHebb always
        with torch.no_grad():
            if isinstance(batch, Mapping):
                out_ng, cache = model(return_cache=True, **b)
            else:
                out_ng, cache = model(x, return_cache=True)

        if not isinstance(cache, Mapping) or "block_inputs" not in cache:
            raise RuntimeError("KP expects cache['block_inputs'] from model(..., return_cache=True).")
        self._softhebb_trunk_step(blocks, cache["block_inputs"])

        loss_t, stats = self._loss_and_stats(task, out_ng, y) if y is not None else (torch.tensor(0.0, device=device), {})
        out_stats = {"loss": float(loss_t.item()), "phase": 0.0}
        out_stats.update(stats)
        self.global_step += 1
        return out_stats

    # ============================================================
    # Finetune helper
    # ============================================================

    def _finetune_one_step(
        self,
        model,
        task,
        batch,
        device,
        blocks,
        y,
        b: Optional[Mapping[str, Any]],
        x: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        if y is None:
            raise ValueError("KP finetune requires labels (y/labels).")

        # choose params
        if self.bp_scope == "head":
            params = self._collect_head_params(blocks)
        else:
            params = [p for p in model.parameters()]

        self._ensure_bp_optimizer(params)
        if self._bp_optimizer is None:
            raise RuntimeError("KP finetune: BP optimizer not initialized (no params).")

        allow_ids = set(id(p) for p in params)
        old_req = self._set_requires_grad_only(model, allow_ids)

        # forward with grads
        if isinstance(batch, Mapping):
            out_bp = model(**b)  # HF style
        else:
            out_bp = model(x)

        loss, stats = self._loss_and_stats(task, out_bp, y)

        self._bp_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._bp_optimizer.step()

        self._restore_requires_grad(model, old_req)

        # optional: keep SoftHebb trunk during finetune
        if self.continue_softhebb_after_pretrain:
            with torch.no_grad():
                if isinstance(batch, Mapping):
                    out_ng, cache = model(return_cache=True, **b)
                else:
                    out_ng, cache = model(x, return_cache=True)

            if not isinstance(cache, Mapping) or "block_inputs" not in cache:
                raise RuntimeError("KP expects cache['block_inputs'] from model(..., return_cache=True).")
            self._softhebb_trunk_step(blocks, cache["block_inputs"])

        out_stats = {"loss": float(loss.item()), "phase": 1.0}
        out_stats.update(stats)
        self.global_step += 1
        return out_stats
