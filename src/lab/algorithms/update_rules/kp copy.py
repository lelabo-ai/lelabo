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


class KP(UpdateRule):
    """
    Hybrid rule:
      - Output blocks (bspec.is_output=True): updated with standard backprop (heads-only optimizer).
      - Intermediate nn.Linear blocks: updated with *local* Supervised Contrastive Learning (SupCon)
        using a per-block projection head + per-block optimizer, with gradients restricted to
        (that block + its projection).
      - Intermediate nn.Conv2d blocks: kept as SoftHebb (no_grad) as in your original code.

    Requirements:
      - model.get_blocks() -> list of BlockSpec-like objects, each has:
          - name: str
          - module: nn.Module
          - is_output: bool
      - model.forward(..., return_cache=True) -> (out, cache)
      - cache["block_inputs"][block_name] exists for blocks you want to update
      - cache["block_outputs"][block_name] exists (used for conv SoftHebb; linear SupCon recomputes forward)
    """

    def __init__(
        self,
        learning_rate: float = 0.05,   # (used for SoftHebb conv update)
        tau: float = 1.0,              # (used for SoftHebb WTA)
        q: float = 0.5,
        anti_hebb: bool = True,
        # heads (backprop)
        head_lr: float = 3e-4,
        head_optim: str = "ano",  # "adamw" | "sgd" | "ano"
        head_weight_decay: float = 0.0,
        # local SupCon (per intermediate linear block)
        supcon_tau: float = 0.1,
        local_lr: float = 3e-4,
        local_optim: str = "adamw",  # "adamw" | "sgd"
        local_weight_decay: float = 0.0,
        proj_dim: int = 128,
        proj_hidden_dim: Optional[int] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        # --- SoftHebb bits (conv)
        self.lr = float(learning_rate)
        self.tau = float(tau)
        self.q = float(q)
        self.anti_hebb = bool(anti_hebb)
        self.eps = float(eps)

        # --- heads (global supervised)
        self.head_lr = float(head_lr)
        self.head_optim = str(head_optim).lower()
        self.head_weight_decay = float(head_weight_decay)
        self._head_optimizer: Optional[torch.optim.Optimizer] = None
        self._head_param_ids: Optional[Tuple[int, ...]] = None

        # --- local SupCon
        self.supcon_tau = float(supcon_tau)
        self.local_lr = float(local_lr)
        self.local_optim = str(local_optim).lower()
        self.local_weight_decay = float(local_weight_decay)
        self.proj_dim = int(proj_dim)
        self.proj_hidden_dim = proj_hidden_dim

        # projection heads per block (registered so they move with .to(...) if needed)
        self._proj_by_name = nn.ModuleDict()
        # per-block local optimizers (Python dict is fine)
        self._local_opt_by_name: Dict[str, torch.optim.Optimizer] = {}
        # remember which param ids were used to build each optimizer
        self._local_param_ids_by_name: Dict[str, Tuple[int, ...]] = {}

        # one-time freeze / param index (for heads)
        self._param_by_id: Optional[Dict[int, nn.Parameter]] = None
        self._frozen_once: bool = True
        self._trainable_ids: set[int] = set()

    # ============================================================
    # SoftHebb core (conv kept)
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
        S = y.new_full(y.shape, -1.0)
        idx = y.argmax(dim=1, keepdim=True)
        S.scatter_(1, idx, 1.0)
        return S

    @torch.no_grad()
    def _update_conv2d_softhebb(self, layer: nn.Conv2d, x: torch.Tensor, u: torch.Tensor):
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

        eta_k = self._eta_per_unit_from_weight(W_flat)
        W_flat.add_(eta_k[:, None] * dW_flat)
        layer.weight.copy_(W_flat.view_as(layer.weight))

        if layer.bias is not None:
            mean_y = y_eff.mean(dim=(0, 2))
            db = mean_y - a * layer.bias
            layer.bias.add_(eta_k * db)

    # ============================================================
    # SupCon (local per Linear block)
    # ============================================================

    def supervised_contrastive_loss(
        self,
        z: torch.Tensor,      # [B, D]
        labels: torch.Tensor, # [B]
        tau: float = 0.1,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Standard SupCon loss (Khosla et al.). Expects labels to be class ids.
        """
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

    def _set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def _make_projection(self, in_dim: int, device: torch.device) -> nn.Module:
        """
        SimCLR/SupCon-style projection head: MLP(in_dim -> hidden -> out_dim)
        """
        out_dim = min(self.proj_dim, in_dim) if self.proj_dim > 0 else in_dim
        hidden = self.proj_hidden_dim if self.proj_hidden_dim is not None else in_dim

        if out_dim == in_dim and hidden == in_dim:
            # still keep a tiny MLP for stability, unless you explicitly want Identity
            pass

        proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        return proj.to(device)

    def _ensure_local_supcon_modules(
        self,
        name: str,
        layer: nn.Linear,
        device: torch.device,
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Creates (or reuses) a projection head + optimizer for this block.
        Optimizer updates BOTH the layer and the projection head (local BP).
        """
        if name not in self._proj_by_name:
            self._proj_by_name[name] = self._make_projection(layer.out_features, device)

        proj = self._proj_by_name[name]

        # Ensure optimizer exists and matches current params (in case model rebuilt)
        params = list(layer.parameters()) + list(proj.parameters())
        param_ids = tuple(sorted(id(p) for p in params))
        if (
            name in self._local_opt_by_name
            and self._local_param_ids_by_name.get(name, None) == param_ids
        ):
            return proj, self._local_opt_by_name[name]

        self._local_param_ids_by_name[name] = param_ids

        if self.local_optim == "sgd":
            opt = torch.optim.SGD(
                params,
                lr=self.local_lr,
                weight_decay=self.local_weight_decay,
                momentum=0.9,
            )
        else:
            opt = torch.optim.AdamW(
                params,
                lr=self.local_lr,
                weight_decay=self.local_weight_decay,
            )

        self._local_opt_by_name[name] = opt
        return proj, opt

    def _local_supcon_update_linear(
        self,
        layer: nn.Linear,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,      # [B, Din]
        labels: torch.Tensor,   # [B]
    ) -> float:
        """
        Local SupCon update for ONE linear layer.
        Important: we temporarily enable grads for this layer (it is frozen globally for head BP efficiency).
        """
        # Temporarily unfreeze this layer (heads-only freeze is kept for global forward)
        was_req = [p.requires_grad for p in layer.parameters()]
        self._set_requires_grad(layer, True)

        # proj is owned by this UpdateRule, keep it trainable
        self._set_requires_grad(proj, True)

        # Local forward + SupCon
        h = layer(xin)          # [B, Dh]
        z = proj(h)             # [B, Dz]
        loss = self.supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Restore layer requires_grad to previous state (usually False)
        for p, prev in zip(layer.parameters(), was_req):
            p.requires_grad = prev

        return float(loss.detach().item())

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
    # One-time freeze (avoid per-step requires_grad toggling) for heads
    # ============================================================

    def _ensure_param_index(self, model: nn.Module) -> None:
        if self._param_by_id is None:
            self._param_by_id = {id(p): p for p in model.parameters()}
            self._frozen_once = False
            self._trainable_ids = set()

    def _freeze_non_head_params_once(self, model: nn.Module, head_params: Sequence[nn.Parameter]) -> None:
        """
        Permanently freezes all params except heads (to make head BP cheap).
        Local SupCon will temporarily unfreeze one intermediate block at a time.
        """
        self._ensure_param_index(model)

        new_ids = {id(p) for p in head_params}

        if not self._frozen_once:
            # Freeze everything once
            for p in self._param_by_id.values():
                p.requires_grad = False

            # Unfreeze heads
            for pid in new_ids:
                p = self._param_by_id.get(pid, None) if self._param_by_id is not None else None
                if p is not None:
                    p.requires_grad = True

            self._trainable_ids = set(new_ids)
            self._frozen_once = True
            return

        # If head set changes, only update differences (cheap)
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

        # HF outputs case
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

    def train_step(self, model, task, batch, device, state=None) -> Dict[str, float]:
        model.train()

        if not hasattr(model, "get_blocks"):
            raise RuntimeError("KP requires models to expose get_blocks() (no fallback).")

        blocks = model.get_blocks()
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise RuntimeError("KP expects model.get_blocks() to return a non-empty list.")

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

        # --- Forward for head (optional) + cache
        can_do_head_bp = (self._head_optimizer is not None) and (y is not None)

        if can_do_head_bp:
            # grads ON; only heads have requires_grad=True (others frozen)
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
            # no supervised head update; still need cache
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
            raise RuntimeError("KP expects cache['block_inputs'] from model(..., return_cache=True).")

        block_inputs: Mapping[str, Any] = cache["block_inputs"]

        # ============================================================
        # Local updates for intermediate blocks
        # ============================================================
        local_supcon_losses: List[float] = []

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

            # --- Linear blocks: local SupCon (needs labels)
            if isinstance(mod, nn.Linear):
                if y is None or (not torch.is_tensor(y)):
                    continue
                if xin.dim() != 2:
                    continue

                # ensure projection + optimizer for this block
                proj, opt = self._ensure_local_supcon_modules(name, mod, device=device)

                # local update (enables grad only for this layer + proj)
                loss_local = self._local_supcon_update_linear(mod, proj, opt, xin, y)
                local_supcon_losses.append(loss_local)

            # --- Conv blocks: keep your SoftHebb update (needs u from cache)
            elif isinstance(mod, nn.Conv2d):
                if xin.dim() != 4:
                    continue
                if "block_outputs" not in cache or name not in cache["block_outputs"]:
                    continue
                u = cache["block_outputs"][name]
                if not torch.is_tensor(u) or u.dim() != 4:
                    continue
                self._update_conv2d_softhebb(mod, xin, u)

            else:
                continue

        out_stats = dict(stats)
        out_stats["loss"] = float(loss_t.item()) if loss_t is not None else 0.0
        if local_supcon_losses:
            out_stats["supcon_loss"] = float(sum(local_supcon_losses) / len(local_supcon_losses))
        else:
            out_stats["supcon_loss"] = 0.0

        self.global_step += 1
        return out_stats
