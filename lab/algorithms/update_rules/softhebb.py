# lab/algorithms/update_rules/softhebb.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule
from .helpers import to_device, extract_loss_and_stats


class SoftHebb(UpdateRule):
    """
    SoftHebb with 2-phase schedule (supervised):
      Phase A (pretrain trunk): pure SoftHebb on the trunk only (no head-loss training).
        - Switch criterion: stagnation of "sum of eta (lr) per unit" over trunk params.
      Phase B (finetune): backprop training "on top" (default: head only), optionally keep SoftHebb on trunk.

    RL path stays the same as before:
      - BP only on heads (actor/critic or q_head), then SoftHebb on the trunk.
    """

    def __init__(
        self,
        learning_rate=0.05,
        tau=1.0,
        q=0.5,
        anti_hebb=True,
        # supervised head update (no autograd) - kept for compatibility
        train_head=True,
        head_lr=0.05,
        # RL hybrid: BP on heads only
        rl_head_backprop=True,
        rl_head_optim="adamw",  # "adamw" | "sgd"
        rl_head_weight_decay=0.0,
        # --- NEW: trunk pretrain schedule (supervised only) ---
        trunk_pretrain=True,
        trunk_pretrain_min_steps=200,
        trunk_pretrain_patience=50,
        trunk_pretrain_rel_tol=1e-3,
        trunk_pretrain_ema_beta=0.05,
        trunk_pretrain_max_steps=10_000,
        # --- NEW: backprop after pretrain (supervised only) ---
        bp_after_pretrain=True,
        bp_scope="head",  # "head" | "all"
        bp_lr=1e-3,
        bp_optim="adamw",  # "adamw" | "sgd"
        bp_weight_decay=0.0,
        # keep SoftHebb trunk updates during BP phase (slower). default False = BP only
        continue_softhebb_after_pretrain=False,
        eps=1e-8,
    ):
        super().__init__()
        self.lr = float(learning_rate)
        self.tau = float(tau)
        self.q = float(q)
        self.anti_hebb = bool(anti_hebb)

        self.train_head = bool(train_head)
        self.head_lr = float(head_lr)

        self.rl_head_backprop = bool(rl_head_backprop)
        self.rl_head_optim = str(rl_head_optim).lower()
        self.rl_head_weight_decay = float(rl_head_weight_decay)

        # schedule
        self.trunk_pretrain = bool(trunk_pretrain)
        self.trunk_pretrain_min_steps = int(trunk_pretrain_min_steps)
        self.trunk_pretrain_patience = int(trunk_pretrain_patience)
        self.trunk_pretrain_rel_tol = float(trunk_pretrain_rel_tol)
        self.trunk_pretrain_ema_beta = float(trunk_pretrain_ema_beta)
        self.trunk_pretrain_max_steps = int(trunk_pretrain_max_steps)

        self.bp_after_pretrain = bool(bp_after_pretrain)
        self.bp_scope = str(bp_scope).lower()
        self.bp_lr = float(bp_lr)
        self.bp_optim = str(bp_optim).lower()
        self.bp_weight_decay = float(bp_weight_decay)
        self.continue_softhebb_after_pretrain = bool(continue_softhebb_after_pretrain)

        self.eps = float(eps)

        # lazily built head optimizer for RL
        self._rl_head_optimizer: Optional[torch.optim.Optimizer] = None
        self._rl_head_param_ids: Optional[Tuple[int, ...]] = None

        # lazily built optimizer for supervised BP
        self._bp_optimizer: Optional[torch.optim.Optimizer] = None
        self._bp_param_ids: Optional[Tuple[int, ...]] = None

        # phase state (supervised only)
        self._phase: str = "pretrain" if self.trunk_pretrain else "normal"
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
    def _update_linear_softhebb(self, layer: nn.Linear, x: torch.Tensor, u: torch.Tensor):
        # x: [B,D], u: [B,K]
        B = x.size(0)
        W = layer.weight  # [K,D]

        y = self._soft_wta(u)  # [B,K]
        y_eff = y * self._winner_sign(y) if self.anti_hebb else y

        term1 = (y_eff.T @ x) / float(B)  # [K,D]
        a = (y_eff * u).mean(dim=0)  # [K]
        dW = term1 - a[:, None] * W  # [K,D]

        eta_k = self._eta_per_unit_from_weight(W)
        layer.weight.add_(eta_k[:, None] * dW)

        if layer.bias is not None:
            mean_y = y_eff.mean(dim=0)
            db = mean_y - a * layer.bias
            layer.bias.add_(eta_k * db)

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

    @torch.no_grad()
    def _update_head_supervised_ce(self, head: nn.Linear, a: torch.Tensor, logits: torch.Tensor, y: torch.Tensor):
        """
        Update CE head (Linear) sans autograd (legacy option):
          dlogits = (softmax(logits) - onehot) / B
          W -= lr * dW
        """
        B = a.size(0)
        probs = F.softmax(logits, dim=1)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, y.view(-1, 1), 1.0)

        dlogits = (probs - onehot) / float(B)
        dW = dlogits.T @ a
        head.weight.add_(-self.head_lr * dW)

        if head.bias is not None:
            db = dlogits.sum(dim=0)
            head.bias.add_(-self.head_lr * db)

    # ============================================================
    # Heads detection / RL heads-only optimizer
    # ============================================================

    def _detect_heads(self, model: nn.Module, out: Any) -> Set[nn.Module]:
        """
        Retourne un set de modules considérés comme "heads" pour RL.
        Priorité:
          - actor_head/critic_head (ActorCriticDiscrete)
          - q_head (QNet)
        """
        heads: Set[nn.Module] = set()

        if isinstance(out, Mapping) and ("logits" in out) and ("value" in out):
            if hasattr(model, "actor_head") and isinstance(model.actor_head, nn.Linear):
                heads.add(model.actor_head)
            if hasattr(model, "critic_head") and isinstance(model.critic_head, nn.Linear):
                heads.add(model.critic_head)
            return heads

        if torch.is_tensor(out):
            if hasattr(model, "q_head") and isinstance(model.q_head, nn.Linear):
                heads.add(model.q_head)
            return heads

        return heads

    def _detect_supervised_head(
        self, model: nn.Module, layer_cache: List[Dict[str, Any]], logits: torch.Tensor
    ) -> Optional[nn.Linear]:
        """
        Pour supervised: essaye de trouver une head Linear stable.
        Priorité:
          - model.linears[-1]
          - model.fc
          - model.classifier
          - sinon: dernière Linear capturée dont u.shape == logits.shape
        """
        if hasattr(model, "linears"):
            try:
                last = list(model.linears)[-1]
                if isinstance(last, nn.Linear):
                    return last
            except Exception:
                pass

        for name in ("fc", "classifier"):
            m = getattr(model, name, None)
            if isinstance(m, nn.Linear):
                return m

        for e in reversed(layer_cache):
            mod = e["module"]
            if isinstance(mod, nn.Linear) and e["u"].shape == logits.shape:
                return mod
        return None

    def _ensure_rl_head_optimizer(self, head_modules: Set[nn.Module], lr: float):
        params = []
        seen = set()
        for m in head_modules:
            for p in m.parameters(recurse=False):
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        if not params:
            self._rl_head_optimizer = None
            self._rl_head_param_ids = None
            return

        param_ids = tuple(sorted(id(p) for p in params))
        if self._rl_head_optimizer is not None and self._rl_head_param_ids == param_ids:
            return

        self._rl_head_param_ids = param_ids

        if self.rl_head_optim == "sgd":
            self._rl_head_optimizer = torch.optim.SGD(params, lr=lr, weight_decay=self.rl_head_weight_decay)
        else:
            self._rl_head_optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.rl_head_weight_decay)

    def _ensure_bp_optimizer(self, params: List[torch.nn.Parameter], lr: float):
        if not params:
            self._bp_optimizer = None
            self._bp_param_ids = None
            return

        param_ids = tuple(sorted(id(p) for p in params))
        if self._bp_optimizer is not None and self._bp_param_ids == param_ids:
            return

        self._bp_param_ids = param_ids
        if self.bp_optim == "sgd":
            self._bp_optimizer = torch.optim.SGD(params, lr=lr, weight_decay=self.bp_weight_decay)
        else:
            self._bp_optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.bp_weight_decay)

    def _set_requires_grad_except(self, model: nn.Module, allow_param_ids: Set[int]) -> Dict[int, bool]:
        old: Dict[int, bool] = {}
        for p in model.parameters():
            old[id(p)] = p.requires_grad
            p.requires_grad = (id(p) in allow_param_ids)
        return old

    def _restore_requires_grad(self, model: nn.Module, old: Dict[int, bool]):
        for p in model.parameters():
            p.requires_grad = old.get(id(p), p.requires_grad)

    # ============================================================
    # Capturing x/u (hooks) - fallback universal
    # ============================================================

    def _capture_xu_hooks(self, model: nn.Module):
        """
        Capture Conv2d (4D) + Linear (2D) : (x,u) au niveau "pré-activation"
        Retourne (layer_cache, hooks)
        """
        layer_cache: List[Dict[str, Any]] = []
        hooks: List[Any] = []

        def hook_fn(module, inputs, output):
            x = inputs[0]
            u = output
            if isinstance(module, nn.Conv2d) and x.dim() == 4 and u.dim() == 4:
                layer_cache.append({"module": module, "x": x.detach(), "u": u.detach()})
            elif isinstance(module, nn.Linear) and x.dim() == 2 and u.dim() == 2:
                layer_cache.append({"module": module, "x": x.detach(), "u": u.detach()})

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                hooks.append(m.register_forward_hook(hook_fn))

        return layer_cache, hooks

    # ============================================================
    # Pretrain switching criterion: stagnation of sum eta
    # ============================================================

    @torch.no_grad()
    def _sum_eta_over_modules(self, modules: Set[nn.Module]) -> float:
        total = 0.0
        for m in modules:
            if isinstance(m, nn.Linear):
                eta = self._eta_per_unit_from_weight(m.weight)
                total += float(eta.sum().item())
            elif isinstance(m, nn.Conv2d):
                W_flat = m.weight.view(m.weight.size(0), -1)
                eta = self._eta_per_unit_from_weight(W_flat)
                total += float(eta.sum().item())
        return total

    def _maybe_switch_phase(self, sum_eta: float) -> bool:
        """
        Updates EMA + stall counter, returns True if we just switched to finetune.
        """
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
    # main API
    # ============================================================

    def train_step(self, model, task, batch, device):
        model.train()

        # -----------------------------
        # CASE 1: HF dict batch (supervised)
        # -----------------------------
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y_true = b.get("labels", None)

            # ---- Supervised pretrain phase: trunk-only SoftHebb, no head loss training
            if self._phase == "pretrain":
                layer_cache, hooks = self._capture_xu_hooks(model)
                with torch.no_grad():
                    outputs = model(**b)
                    logits = outputs.logits
                for h in hooks:
                    h.remove()

                head = self._detect_supervised_head(model, layer_cache, logits)

                # modules to update: trunk only (exclude detected head)
                trunk_modules: Set[nn.Module] = set()
                for e in layer_cache:
                    mod = e["module"]
                    if (head is not None) and (mod is head):
                        continue
                    trunk_modules.add(mod)

                with torch.no_grad():
                    for e in layer_cache:
                        mod = e["module"]
                        if mod not in trunk_modules:
                            continue
                        x_m, u_m = e["x"], e["u"]
                        if isinstance(mod, nn.Conv2d):
                            self._update_conv2d_softhebb(mod, x_m, u_m)
                        elif isinstance(mod, nn.Linear):
                            self._update_linear_softhebb(mod, x_m, u_m)

                # stagnation criterion
                self._pretrain_steps += 1
                sum_eta = self._sum_eta_over_modules(trunk_modules)
                just_switched = self._maybe_switch_phase(sum_eta)

                stats: Dict[str, float] = {"loss": 0.0}
                if y_true is not None:
                    loss = task.loss(logits, y_true)
                    stats["loss"] = float(loss.item())
                    if torch.is_tensor(y_true) and y_true.dtype in (torch.int64, torch.int32, torch.int16):
                        preds = logits.argmax(dim=-1)
                        stats["acc"] = float((preds == y_true).float().mean().item())

                stats["phase"] = 0.0  # 0=pretrain, 1=finetune (keeps logs numeric-friendly)
                stats["sum_eta"] = float(sum_eta)
                stats["eta_ema"] = float(self._eta_ema if self._eta_ema is not None else sum_eta)
                stats["eta_stall"] = float(self._stall_steps)
                stats["switched"] = 1.0 if just_switched else 0.0

                self.global_step += 1
                return stats

            # ---- Supervised finetune: backprop "on top" (default head), optionally keep SoftHebb trunk
            if self._phase == "finetune" and self.bp_after_pretrain:
                # we can (optionally) capture x/u during the BP forward (no extra forward)
                layer_cache: List[Dict[str, Any]] = []
                hooks: List[Any] = []
                if self.continue_softhebb_after_pretrain:
                    layer_cache, hooks = self._capture_xu_hooks(model)

                outputs = model(**b)
                logits = outputs.logits

                for h in hooks:
                    h.remove()

                if y_true is None:
                    raise ValueError("SoftHebb finetune (HF dict): expected 'labels' in batch for supervised BP.")

                # detect head and build BP param list
                head = None
                try:
                    # if hooks were enabled, layer_cache exists; else pass empty
                    head = self._detect_supervised_head(model, layer_cache, logits)
                except Exception:
                    head = None

                bp_params: List[torch.nn.Parameter] = []
                if self.bp_scope == "head" and isinstance(head, nn.Module):
                    bp_params = list(head.parameters(recurse=False))
                else:
                    bp_params = [p for p in model.parameters()]

                self._ensure_bp_optimizer(bp_params, lr=self.bp_lr)
                if self._bp_optimizer is None:
                    raise RuntimeError("SoftHebb finetune: BP optimizer not initialized (no params).")

                # (optional) freeze non-BP params to go faster
                allow_ids = set(id(p) for p in bp_params)
                old_req = self._set_requires_grad_except(model, allow_ids)

                loss_res = task.loss(logits, y_true)
                loss, extra = extract_loss_and_stats(loss_res)

                self._bp_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._bp_optimizer.step()

                self._restore_requires_grad(model, old_req)

                # optional: keep SoftHebb trunk updates during finetune
                if self.continue_softhebb_after_pretrain and layer_cache:
                    trunk_modules: Set[nn.Module] = set()
                    for e in layer_cache:
                        mod = e["module"]
                        if (head is not None) and (mod is head):
                            continue
                        trunk_modules.add(mod)

                    with torch.no_grad():
                        for e in layer_cache:
                            mod = e["module"]
                            if mod not in trunk_modules:
                                continue
                            x_m, u_m = e["x"], e["u"]
                            if isinstance(mod, nn.Conv2d):
                                self._update_conv2d_softhebb(mod, x_m, u_m)
                            elif isinstance(mod, nn.Linear):
                                self._update_linear_softhebb(mod, x_m, u_m)

                stats = {"loss": float(loss.item())}
                stats.update(extra)

                # quick acc if classification
                if torch.is_tensor(y_true) and y_true.dtype in (torch.int64, torch.int32, torch.int16):
                    preds = logits.argmax(dim=-1)
                    stats["acc"] = float((preds == y_true).float().mean().item())

                stats["phase"] = 1.0
                self.global_step += 1
                return stats

            # ---- Legacy path (original behavior): pure no_grad SoftHebb + optional CE head update
            layer_cache, hooks = self._capture_xu_hooks(model)
            with torch.no_grad():
                outputs = model(**b)
                logits = outputs.logits
            for h in hooks:
                h.remove()

            head = self._detect_supervised_head(model, layer_cache, logits)

            with torch.no_grad():
                for e in layer_cache:
                    mod = e["module"]
                    x_m, u_m = e["x"], e["u"]
                    if isinstance(mod, nn.Conv2d):
                        self._update_conv2d_softhebb(mod, x_m, u_m)
                    elif isinstance(mod, nn.Linear):
                        if (head is not None) and (mod is head) and self.train_head and (y_true is not None):
                            self._update_head_supervised_ce(mod, x_m, logits, y_true)
                        else:
                            self._update_linear_softhebb(mod, x_m, u_m)

            stats = {"loss": 0.0}
            if y_true is not None:
                loss = task.loss(logits, y_true)
                stats["loss"] = float(loss.item())
                if torch.is_tensor(y_true) and y_true.dtype in (torch.int64, torch.int32, torch.int16):
                    preds = logits.argmax(dim=-1)
                    stats["acc"] = float((preds == y_true).float().mean().item())

            # phase logging (normal)
            stats["phase"] = 1.0 if self._phase == "finetune" else 0.0
            self.global_step += 1
            return stats

        # -----------------------------
        # CASE 2: tuple batch (x,y) supervised OR RL (y dict)
        # -----------------------------
        x, y = batch
        x = to_device(x, device)

        is_rl = isinstance(y, Mapping)
        y = to_device(y, device)

        # ---- 1) capture caches with a no_grad forward (used by RL always; by supervised pretrain/legacy; by finetune only if continuing SoftHebb)
        # We'll handle per-path below to avoid unnecessary overhead.

        # -----------------------------
        # RL path: BP on heads only + SoftHebb trunk
        # -----------------------------
        if is_rl:
            if not self.rl_head_backprop:
                raise NotImplementedError("SoftHebb: RL batch détecté mais rl_head_backprop=False.")

            # capture for trunk SoftHebb
            layer_cache, hooks = self._capture_xu_hooks(model)
            with torch.no_grad():
                out0 = model(x)
            for h in hooks:
                h.remove()

            head_modules = self._detect_heads(model, out0)
            if not head_modules:
                raise RuntimeError(
                    "SoftHebb RL: impossible de détecter les heads. "
                    "Ajoute actor_head/critic_head (actor-critic) ou q_head (DQN) dans le modèle."
                )

            self._ensure_rl_head_optimizer(head_modules, lr=self.head_lr)
            if self._rl_head_optimizer is None:
                raise RuntimeError("SoftHebb RL: optimizer heads non initialisé.")

            allow_ids: Set[int] = set()
            for m in head_modules:
                for p in m.parameters(recurse=False):
                    allow_ids.add(id(p))
            old_req = self._set_requires_grad_except(model, allow_ids)

            out = model(x)
            loss_res = task.loss(out, y)
            loss, extra = extract_loss_and_stats(loss_res)

            self._rl_head_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._rl_head_optimizer.step()

            self._restore_requires_grad(model, old_req)

            with torch.no_grad():
                for e in layer_cache:
                    mod = e["module"]
                    if mod in head_modules:
                        continue
                    x_m, u_m = e["x"], e["u"]
                    if isinstance(mod, nn.Conv2d):
                        self._update_conv2d_softhebb(mod, x_m, u_m)
                    elif isinstance(mod, nn.Linear):
                        self._update_linear_softhebb(mod, x_m, u_m)

            stats = {"loss": float(loss.item())}
            stats.update(extra)
            self.global_step += 1
            return stats

        # -----------------------------
        # Supervised tuple path:
        #   - pretrain: trunk-only SoftHebb (fast, no head loss training)
        #   - finetune: BP on head/all
        #   - legacy: SoftHebb + optional CE head update
        # -----------------------------

        # ---- PRETRAIN
        if self._phase == "pretrain":
            layer_cache, hooks = self._capture_xu_hooks(model)
            with torch.no_grad():
                out0 = model(x)
            for h in hooks:
                h.remove()

            logits = out0.get("logits", None) if isinstance(out0, Mapping) else out0
            if logits is None or not torch.is_tensor(logits):
                raise ValueError("SoftHebb supervised pretrain: model output must be a logits tensor (or dict with 'logits').")
            if not torch.is_tensor(y):
                raise TypeError("SoftHebb supervised pretrain: y must be a tensor (labels).")

            head = self._detect_supervised_head(model, layer_cache, logits)

            trunk_modules: Set[nn.Module] = set()
            for e in layer_cache:
                mod = e["module"]
                if (head is not None) and (mod is head):
                    continue
                trunk_modules.add(mod)

            with torch.no_grad():
                for e in layer_cache:
                    mod = e["module"]
                    if mod not in trunk_modules:
                        continue
                    x_m, u_m = e["x"], e["u"]
                    if isinstance(mod, nn.Conv2d):
                        self._update_conv2d_softhebb(mod, x_m, u_m)
                    elif isinstance(mod, nn.Linear):
                        self._update_linear_softhebb(mod, x_m, u_m)

            self._pretrain_steps += 1
            sum_eta = self._sum_eta_over_modules(trunk_modules)
            just_switched = self._maybe_switch_phase(sum_eta)

            loss = task.loss(logits, y)
            stats = {"loss": float(loss.item())}
            if hasattr(task, "metrics"):
                try:
                    stats.update(task.metrics(logits, y))
                except Exception:
                    pass

            stats["phase"] = 0.0
            stats["sum_eta"] = float(sum_eta)
            stats["eta_ema"] = float(self._eta_ema if self._eta_ema is not None else sum_eta)
            stats["eta_stall"] = float(self._stall_steps)
            stats["switched"] = 1.0 if just_switched else 0.0

            self.global_step += 1
            if (self.global_step % 100) == 0 or just_switched:
                    print(
                        f"[SoftHebb] step={self.global_step} phase={self._phase} "
                        f"pretrain_steps={self._pretrain_steps} sum_eta={sum_eta:.4e} "
                        f"eta_ema={(self._eta_ema if self._eta_ema is not None else sum_eta):.4e} "
                        f"stall={self._stall_steps} switched={just_switched}"
                    )

            return stats

        # ---- FINETUNE (BP)
        if self._phase == "finetune" and self.bp_after_pretrain:
            layer_cache: List[Dict[str, Any]] = []
            hooks: List[Any] = []
            if self.continue_softhebb_after_pretrain:
                layer_cache, hooks = self._capture_xu_hooks(model)

            out = model(x)

            for h in hooks:
                h.remove()

            logits = out.get("logits", None) if isinstance(out, Mapping) else out
            if logits is None or not torch.is_tensor(logits):
                raise ValueError("SoftHebb supervised finetune: model output must be a logits tensor (or dict with 'logits').")
            if not torch.is_tensor(y):
                raise TypeError("SoftHebb supervised finetune: y must be a tensor (labels).")

            head = None
            if self.bp_scope == "head":
                try:
                    head = self._detect_supervised_head(model, layer_cache, logits) if layer_cache else None
                except Exception:
                    head = None

            if self.bp_scope == "head" and isinstance(head, nn.Module):
                bp_params = list(head.parameters(recurse=False))
            else:
                bp_params = [p for p in model.parameters()]

            self._ensure_bp_optimizer(bp_params, lr=self.bp_lr)
            if self._bp_optimizer is None:
                raise RuntimeError("SoftHebb finetune: BP optimizer not initialized (no params).")

            allow_ids = set(id(p) for p in bp_params)
            old_req = self._set_requires_grad_except(model, allow_ids)

            loss_res = task.loss(logits, y)
            loss, extra = extract_loss_and_stats(loss_res)

            self._bp_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._bp_optimizer.step()

            self._restore_requires_grad(model, old_req)

            if self.continue_softhebb_after_pretrain and layer_cache:
                trunk_modules: Set[nn.Module] = set()
                for e in layer_cache:
                    mod = e["module"]
                    if (head is not None) and (mod is head):
                        continue
                    trunk_modules.add(mod)

                with torch.no_grad():
                    for e in layer_cache:
                        mod = e["module"]
                        if mod not in trunk_modules:
                            continue
                        x_m, u_m = e["x"], e["u"]
                        if isinstance(mod, nn.Conv2d):
                            self._update_conv2d_softhebb(mod, x_m, u_m)
                        elif isinstance(mod, nn.Linear):
                            self._update_linear_softhebb(mod, x_m, u_m)

            stats = {"loss": float(loss.item())}
            stats.update(extra)
            stats["phase"] = 1.0
            self.global_step += 1
            return stats

        # ---- LEGACY supervised behavior (original): one no_grad forward + SoftHebb everywhere + optional CE head update
        layer_cache, hooks = self._capture_xu_hooks(model)
        with torch.no_grad():
            out0 = model(x)
        for h in hooks:
            h.remove()

        logits = out0.get("logits", None) if isinstance(out0, Mapping) else out0
        if logits is None or not torch.is_tensor(logits):
            raise ValueError("SoftHebb supervised: model output must be a logits tensor (or dict with 'logits').")
        if not torch.is_tensor(y):
            raise TypeError("SoftHebb supervised: y must be a tensor (labels).")

        head = self._detect_supervised_head(model, layer_cache, logits)

        with torch.no_grad():
            for e in layer_cache:
                mod = e["module"]
                x_m, u_m = e["x"], e["u"]
                if isinstance(mod, nn.Conv2d):
                    self._update_conv2d_softhebb(mod, x_m, u_m)
                elif isinstance(mod, nn.Linear):
                    if (head is not None) and (mod is head) and self.train_head:
                        self._update_head_supervised_ce(mod, x_m, logits, y)
                    else:
                        self._update_linear_softhebb(mod, x_m, u_m)

        loss = task.loss(logits, y)
        stats = {"loss": float(loss.item())}
        if hasattr(task, "metrics"):
            try:
                stats.update(task.metrics(logits, y))
            except Exception:
                pass

        stats["phase"] = 1.0 if self._phase == "finetune" else 0.0
        self.global_step += 1
        return stats
