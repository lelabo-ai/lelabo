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
    SoftHebb:
      - Supervised: SoftHebb sur tout + (optionnel) update CE sur la head
      - RL (PPO/DQN-like): BP uniquement sur les heads (actor/critic ou q_head),
        puis SoftHebb sur le reste.

    Fonctionne sur:
      - MLP / QNet / ActorCritic (avec heads explicites si tu as suivi le ménage)
      - CNN/ResNet (fallback hooks)
      - HF dict batch: on fait SoftHebb sur les Linear/Conv2d 2D/4D capturés (souvent surtout head)
    """

    def __init__(
        self,
        learning_rate=0.05,
        tau=1.0,
        q=0.5,
        anti_hebb=True,
        # supervised head update (no autograd)
        train_head=True,
        head_lr=0.05,
        # RL hybrid: BP on heads only
        rl_head_backprop=True,
        rl_head_optim="ano",  # "adamw" | "sgd"
        rl_head_weight_decay=0.0,
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

        self.eps = float(eps)

        # lazily built head optimizer for RL
        self._rl_head_optimizer: Optional[torch.optim.Optimizer] = None
        self._rl_head_param_ids: Optional[Tuple[int, ...]] = None

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
        Update CE head (Linear) sans autograd:
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
            # actor-critic
            if hasattr(model, "actor_head") and isinstance(model.actor_head, nn.Linear):
                heads.add(model.actor_head)
            if hasattr(model, "critic_head") and isinstance(model.critic_head, nn.Linear):
                heads.add(model.critic_head)
            return heads

        # DQN-style
        if torch.is_tensor(out):
            if hasattr(model, "q_head") and isinstance(model.q_head, nn.Linear):
                heads.add(model.q_head)
            return heads

        return heads

    def _detect_supervised_head(self, model: nn.Module, layer_cache: List[Dict[str, Any]], logits: torch.Tensor) -> Optional[nn.Linear]:
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

        # fallback shape match in captured cache
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
        elif self.rl_head_optim == 'ano':
            from ano_optimizer import Ano
            self._rl_head_optimizer = Ano(params, lr=lr, weight_decay=self.rl_head_weight_decay)
        else:
            self._rl_head_optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.rl_head_weight_decay)

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
    # main API
    # ============================================================

    def train_step(self, model, task, batch, device):
        model.train()

        # -----------------------------
        # CASE 1: HF dict batch (supervised)
        # -----------------------------
        if isinstance(batch, Mapping):
            b = to_device(batch, device)

            layer_cache, hooks = self._capture_xu_hooks(model)
            with torch.no_grad():
                outputs = model(**b)
                logits = outputs.logits
                y_true = b.get("labels", None)
            for h in hooks:
                h.remove()

            # head detection
            head = self._detect_supervised_head(model, layer_cache, logits)

            # updates (pure no_grad)
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

            # stats
            stats = {"loss": 0.0}
            if y_true is not None:
                loss = task.loss(logits, y_true)
                stats["loss"] = float(loss.item())
                # acc rapide si possible
                if hasattr(outputs, "logits") and torch.is_tensor(y_true) and y_true.dtype in (torch.int64, torch.int32, torch.int16):
                    preds = outputs.logits.argmax(dim=-1)
                    stats["acc"] = float((preds == y_true).float().mean().item())

            self.global_step += 1
            return stats

        # -----------------------------
        # CASE 2: tuple batch (x,y) supervised OR RL (y dict)
        # -----------------------------
        x, y = batch
        x = to_device(x, device)

        is_rl = isinstance(y, Mapping)
        y = to_device(y, device)

        # ---- 1) capture caches with a no_grad forward
        layer_cache, hooks = self._capture_xu_hooks(model)
        with torch.no_grad():
            out0 = model(x)
        for h in hooks:
            h.remove()

        # -----------------------------
        # RL path: BP on heads only + SoftHebb trunk
        # -----------------------------
        if is_rl:
            if not self.rl_head_backprop:
                raise NotImplementedError("SoftHebb: RL batch détecté mais rl_head_backprop=False.")

            head_modules = self._detect_heads(model, out0)
            if not head_modules:
                raise RuntimeError(
                    "SoftHebb RL: impossible de détecter les heads. "
                    "Ajoute actor_head/critic_head (actor-critic) ou q_head (DQN) dans le modèle."
                )

            self._ensure_rl_head_optimizer(head_modules, lr=self.head_lr)
            if self._rl_head_optimizer is None:
                raise RuntimeError("SoftHebb RL: optimizer heads non initialisé.")

            # freeze trunk params (sauve mémoire / évite grads inutiles)
            allow_ids: Set[int] = set()
            for m in head_modules:
                for p in m.parameters(recurse=False):
                    allow_ids.add(id(p))
            old_req = self._set_requires_grad_except(model, allow_ids)

            # forward + autograd (sans hooks) sur heads only
            out = model(x)
            loss_res = task.loss(out, y)
            loss, extra = extract_loss_and_stats(loss_res)

            self._rl_head_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._rl_head_optimizer.step()

            self._restore_requires_grad(model, old_req)

            # SoftHebb trunk update (after optimizer step)
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
        # Supervised path: SoftHebb + optional CE head update
        # -----------------------------
        if isinstance(out0, Mapping):
            # en supervised classique, ton modèle renvoie généralement un tensor logits,
            # mais on supporte dict si tu veux
            logits0 = out0.get("logits", None)
            if logits0 is None or not torch.is_tensor(logits0):
                raise ValueError("SoftHebb supervised: output dict sans 'logits' tensor.")
            logits = logits0
        else:
            logits = out0

        if not torch.is_tensor(y):
            raise TypeError("SoftHebb supervised: y doit être un tensor (labels).")

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

        self.global_step += 1
        return stats
