# algorithms/update_rules/softhebb.py
import torch
import torch.nn.functional as F
from collections.abc import Mapping

from .base import UpdateRule


class SoftHebb(UpdateRule):
    """
    SoftHebb local updates on trunk + (optional) supervised head update (classification).
    For RL (PPO/DQN-like): do Backprop ONLY on heads, SoftHebb on trunk.

    RL batch convention expected:
      train_step(model, task, (x, y_dict), device)
    where y_dict is a mapping (advantages/actions/returns/...)
    and model(x) returns either:
      - dict {"logits": ..., "value": ...}  (actor-critic)
      - tensor Q-values [B,A]              (DQN-style)
    """

    def __init__(
        self,
        learning_rate=0.05,
        tau=1.0,
        q=0.5,
        anti_hebb=True,

        # supervised classification head update (no autograd)
        train_head=True,
        head_lr=0.05,

        # RL hybrid: BP on heads only
        rl_head_backprop=True,
        rl_head_optim="adamw",          # "adamw" | "sgd"
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
        self._rl_head_optimizer = None
        self._rl_head_param_ids = None

    # -------------------------
    # helpers (SoftHebb)
    # -------------------------
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
            idx = y.argmax(dim=1)  # [B,L]
            S.scatter_(1, idx.unsqueeze(1), 1.0)
        return S

    @torch.no_grad()
    def _update_linear_softhebb(self, layer: torch.nn.Linear, x: torch.Tensor, u: torch.Tensor):
        B, _Din = x.shape
        W = layer.weight  # [K,D]

        y = self._soft_wta(u)  # [B,K]
        y_eff = y * self._winner_sign(y) if self.anti_hebb else y

        term1 = (y_eff.T @ x) / float(B)  # [K,D]
        a = (y_eff * u).mean(dim=0)       # [K]
        dW = term1 - a[:, None] * W       # [K,D]

        eta_k = self._eta_per_unit_from_weight(W)
        layer.weight.add_(eta_k[:, None] * dW)

        if layer.bias is not None:
            mean_y = y_eff.mean(dim=0)        # [K]
            db = mean_y - a * layer.bias      # [K]
            layer.bias.add_(eta_k * db)

    @torch.no_grad()
    def _update_conv2d_softhebb(self, layer: torch.nn.Conv2d, x: torch.Tensor, u: torch.Tensor):
        B, _Cin, _Hin, _Win = x.shape
        Cout = u.shape[1]

        patches = F.unfold(
            x,
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )  # [B, D, L]

        L = patches.shape[2]
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
    def _update_head_supervised_ce(self, head: torch.nn.Linear, a: torch.Tensor, logits: torch.Tensor, y: torch.Tensor):
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

    # -------------------------
    # helpers (heads identification + RL head optimizer)
    # -------------------------
    def _match_linear_that_produced(self, layer_cache, target: torch.Tensor):
        if target is None:
            return None

        # best: data_ptr match
        for e in layer_cache:
            if isinstance(e["module"], torch.nn.Linear):
                try:
                    if e["u"].data_ptr() == target.data_ptr():
                        return e["module"]
                except Exception:
                    pass

        # fallback: shape match
        for e in reversed(layer_cache):
            m = e["module"]
            u = e["u"]
            if not isinstance(m, torch.nn.Linear):
                continue
            if u.shape == target.shape:
                return m
            if target.dim() == 1 and u.dim() == 2 and u.shape[0] == target.shape[0] and u.shape[1] == 1:
                return m

        return None

    def _ensure_rl_head_optimizer(self, head_modules, lr: float):
        params = []
        seen = set()
        for m in head_modules:
            if m is None:
                continue
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
            self._rl_head_optimizer = torch.optim.SGD(
                params, lr=lr, weight_decay=self.rl_head_weight_decay
            )
        else:
            self._rl_head_optimizer = torch.optim.AdamW(
                params, lr=lr, weight_decay=self.rl_head_weight_decay
            )

    def _set_requires_grad_except(self, model: torch.nn.Module, allow_param_ids: set[int]):
        old = {}
        for p in model.parameters():
            old[id(p)] = p.requires_grad
            p.requires_grad = (id(p) in allow_param_ids)
        return old

    def _restore_requires_grad(self, model: torch.nn.Module, old: dict):
        for p in model.parameters():
            p.requires_grad = old.get(id(p), p.requires_grad)

    # -------------------------
    # main API
    # -------------------------
    def train_step(self, model, task, batch, device):
        model.train()

        # ---- hooks capture (module, x, u) ----
        layer_cache = []
        hooks = []

        def hook_fn(module, inputs, output):
            x = inputs[0]
            u = output
            if isinstance(module, torch.nn.Conv2d) and x.dim() == 4 and u.dim() == 4:
                layer_cache.append({"module": module, "x": x.detach(), "u": u.detach()})
            elif isinstance(module, torch.nn.Linear) and x.dim() == 2 and u.dim() == 2:
                layer_cache.append({"module": module, "x": x.detach(), "u": u.detach()})

        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(m.register_forward_hook(hook_fn))

        # -------------------------
        # CASE A: Transformers dict batch (supervised)
        # -------------------------
        if isinstance(batch, Mapping):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                y_true = batch.get("labels", None)

            for h in hooks:
                h.remove()

            head_logits_mod = self._match_linear_that_produced(layer_cache, logits)

            # updates
            with torch.no_grad():
                for e in layer_cache:
                    mod = e["module"]
                    x_m, u_m = e["x"], e["u"]

                    if isinstance(mod, torch.nn.Conv2d):
                        self._update_conv2d_softhebb(mod, x_m, u_m)
                    elif isinstance(mod, torch.nn.Linear):
                        is_head = (head_logits_mod is not None and mod is head_logits_mod)
                        if is_head and self.train_head and (y_true is not None):
                            self._update_head_supervised_ce(mod, x_m, logits, y_true)
                        else:
                            self._update_linear_softhebb(mod, x_m, u_m)

            stats = {"loss": 0.0}
            if y_true is not None:
                ce = task.loss(logits, y_true)
                stats["loss"] = float(ce.item())
                if hasattr(task, "metrics"):
                    stats.update(task.metrics(logits, y_true))

            self.global_step += 1
            return stats

        # -------------------------
        # tuple batch: (x, y) where y can be tensor (supervised) or dict (RL)
        # -------------------------
        x, y = batch
        x = x.to(device)

        is_rl = isinstance(y, Mapping)

        # -------------------------
        # 1) First forward ONLY to collect caches + identify heads (no graph)
        # -------------------------
        with torch.no_grad():
            out0 = model(x)
            if isinstance(out0, Mapping):
                logits0 = out0.get("logits", None)
                value0 = out0.get("value", None)
            else:
                logits0 = out0
                value0 = None

        for h in hooks:
            h.remove()

        head_logits_mod = self._match_linear_that_produced(layer_cache, logits0)
        head_value_mod = self._match_linear_that_produced(layer_cache, value0) if value0 is not None else None
        head_modules = {m for m in [head_logits_mod, head_value_mod] if m is not None}

        # -------------------------
        # 2) RL: BP on heads (autograd) then SoftHebb trunk (no_grad)
        # -------------------------
        if is_rl:
            if not self.rl_head_backprop:
                raise NotImplementedError("RL batch détecté mais rl_head_backprop=False.")

            self._ensure_rl_head_optimizer(head_modules, lr=self.head_lr)
            if self._rl_head_optimizer is None:
                raise RuntimeError(
                    "Impossible d'identifier les têtes Linear (logits/value). "
                    "Expose explicitement actor_head/critic_head dans le modèle ou adapte le matching."
                )

            # allow grads only for head params
            allow_ids = set()
            for m in head_modules:
                for p in m.parameters(recurse=False):
                    allow_ids.add(id(p))
            old_req = self._set_requires_grad_except(model, allow_ids)

            # forward with graph (but only heads require grad)
            out = model(x)
            loss_out = task.loss(out, y)

            extra_stats = {}
            if isinstance(loss_out, tuple) and len(loss_out) == 2:
                loss, extra = loss_out
                if isinstance(extra, Mapping):
                    extra_stats.update({k: float(v) for k, v in extra.items() if isinstance(v, (int, float))})
            elif isinstance(loss_out, Mapping):
                loss = loss_out.get("loss", None)
                for k, v in loss_out.items():
                    if k != "loss" and isinstance(v, (int, float)):
                        extra_stats[k] = float(v)
                if loss is None:
                    raise ValueError("task.loss(...) a retourné un dict sans clé 'loss'.")
            else:
                loss = loss_out

            if not torch.is_tensor(loss):
                self._restore_requires_grad(model, old_req)
                raise TypeError("task.loss(...) doit retourner un torch scalar (ou (loss,stats)/dict).")

            self._rl_head_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._rl_head_optimizer.step()

            # restore requires_grad flags
            self._restore_requires_grad(model, old_req)

            # IMPORTANT: SoftHebb updates AFTER backward/step (avoid inplace autograd crash)
            with torch.no_grad():
                for e in layer_cache:
                    mod = e["module"]
                    if mod in head_modules:
                        continue
                    x_m, u_m = e["x"], e["u"]
                    if isinstance(mod, torch.nn.Conv2d):
                        self._update_conv2d_softhebb(mod, x_m, u_m)
                    elif isinstance(mod, torch.nn.Linear):
                        self._update_linear_softhebb(mod, x_m, u_m)

            stats = {"loss": float(loss.item())}
            stats.update(extra_stats)
            self.global_step += 1
            return stats

        # -------------------------
        # 3) Supervised tuple (x, y_tensor): SoftHebb + optional CE head update (no autograd)
        # -------------------------
        y = y.to(device)

        # We already have logits0 (no_grad) and caches
        logits = logits0
        y_true = y

        # apply updates (no_grad)
        with torch.no_grad():
            for e in layer_cache:
                mod = e["module"]
                x_m, u_m = e["x"], e["u"]
                if isinstance(mod, torch.nn.Conv2d):
                    self._update_conv2d_softhebb(mod, x_m, u_m)
                elif isinstance(mod, torch.nn.Linear):
                    is_head = (head_logits_mod is not None and mod is head_logits_mod)
                    if is_head and self.train_head:
                        self._update_head_supervised_ce(mod, x_m, logits, y_true)
                    else:
                        self._update_linear_softhebb(mod, x_m, u_m)

        stats = {}
        ce = task.loss(logits, y_true)
        stats["loss"] = float(ce.item())
        if hasattr(task, "metrics"):
            stats.update(task.metrics(logits, y_true))

        self.global_step += 1
        return stats
