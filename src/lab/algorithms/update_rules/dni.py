# lab/algorithms/update_rules/dni.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import to_device
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


def _parse_loss(loss_out: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compatible avec:
      - loss tensor
      - (loss, stats_dict)
      - {"loss": loss, ...}
    """
    extra: Dict[str, float] = {}
    if torch.is_tensor(loss_out):
        return loss_out, extra
    if isinstance(loss_out, tuple) and len(loss_out) == 2:
        loss, stats = loss_out
        if isinstance(stats, Mapping):
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    extra[k] = float(v)
        return loss, extra
    if isinstance(loss_out, Mapping):
        loss = loss_out.get("loss", None)
        if loss is None or not torch.is_tensor(loss):
            raise ValueError("task.loss(...) a retourné un dict sans 'loss' tensor.")
        for k, v in loss_out.items():
            if k != "loss" and isinstance(v, (int, float)):
                extra[k] = float(v)
        return loss, extra
    raise TypeError("task.loss(...) doit retourner un tensor loss (ou (loss,stats) / dict).")


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
    Decoupled Neural Interfaces (Synthetic Gradients), proche de ton impl.

    ✅ Supervised: identique à avant.
    ✅ RL:
      - ActorCriticDiscrete (actor/critic): DNI séparé sur actor et critic
      - DQN/QNet: DNI sur la chaîne main

    Limite (comme ta version): ne gère que chaînes de nn.Linear 2D.
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
        sg_hidden: int = 0,             # 0 => linear SG model
        condition_on_label: bool = False,  # cDNI (supervised only)
        lambda_mix: float = 0.0,        # BP(λ)
        sg_scale: float = 1.0,
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

        # SG models par "branche" (main/actor/critic)
        self._sg_models: nn.ModuleDict = nn.ModuleDict()
        self._built_for: Dict[str, Tuple[int, ...]] = {}
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

    def _capture_linear_xu_hooks(self, module: nn.Module):
        """
        Capture Linear layers in *execution order*:
          x: input to Linear  [B, Din]  (this is h_{i-1})
          u: output of Linear [B, Dout] (pre-activation)
        """
        cache: List[Dict[str, Any]] = []
        hooks: List[Any] = []

        def hook_fn(m, inputs, output):
            x = inputs[0]
            u = output
            if isinstance(m, nn.Linear) and x.dim() == 2 and u.dim() == 2:
                cache.append({"module": m, "x": x.detach(), "u": u.detach()})

        for m in module.modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(hook_fn))

        return cache, hooks

    # ---------------- optimizers ----------------

    def _ensure_optimizers(self, model: nn.Module):
        # net optimizer (always ok)
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

        # SG optimizer: only create if SG params exist
        if self._sg_optimizer is None:
            sg_params = list(self._sg_models.parameters())
            if len(sg_params) == 0:
                # nothing to optimize yet; we'll create it once SG models are built
                return

            if self.sg_optim == "sgd":
                self._sg_optimizer = torch.optim.SGD(
                    sg_params,
                    lr=self.sg_lr,
                    momentum=0.0,
                    weight_decay=self.sg_weight_decay,
                )
            else:
                self._sg_optimizer = torch.optim.AdamW(
                    sg_params,
                    lr=self.sg_lr,
                    weight_decay=self.sg_weight_decay,
                )


    def _onehot(self, y: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(y.long(), num_classes=num_classes).to(dtype=torch.float32, device=y.device)

    def _maybe_build_sg_models(
        self,
        key: str,
        task: Any,
        y: torch.Tensor,
        linear_cache: List[Dict[str, Any]],
        out_tensor: torch.Tensor,
        device: str,
        allow_condition_on_label: bool,
    ):
        """
        Build one SG model per hidden activation h_i (i=0..N-2),
        i.e. len(linears)-1 models.
        """
        if len(linear_cache) < 2:
            self._sg_models[key] = nn.ModuleList()
            self._built_for[key] = tuple()
            return

        sig: List[int] = []
        for e in linear_cache:
            m: nn.Linear = e["module"]
            sig.extend([int(m.in_features), int(m.out_features)])
        signature = tuple(sig)

        if self._built_for.get(key, None) == signature and len(self._sg_models[key]) == (len(linear_cache) - 1):
            return

        self._built_for[key] = signature
        models = nn.ModuleList()

        cond = self.condition_on_label and allow_condition_on_label
        num_classes = _infer_num_classes(task, y, out_tensor) if cond else 0
        if cond:
            self._num_classes = num_classes

        for i in range(len(linear_cache) - 1):
            dim_hi = int(linear_cache[i]["u"].size(1))
            in_dim = dim_hi + (num_classes if cond else 0)
            models.append(_SGModel(in_dim=in_dim, out_dim=dim_hi, hidden_dim=self.sg_hidden))

        self._sg_models[key] = models.to(device)
        self._sg_optimizer = None  # params changed -> rebuild optimizer

    # ---------------- core DNI on one chain ----------------

    def _dni_chain(
        self,
        key: str,
        linear_cache: List[Dict[str, Any]],
        g_out: torch.Tensor,           # dL/d(last_output)
        y: Optional[torch.Tensor],      # labels only if cDNI enabled
        allow_condition_on_label: bool,
        device: str,
    ) -> torch.Tensor:
        """
        Ecrit les grads .grad sur les Linear de la chaîne.
        Retourne sg_total_loss (différentiable) pour entraîner les SG models.
        """
        if len(linear_cache) == 0:
            return torch.tensor(0.0, device=device)

        N = len(linear_cache)
        u_list = [e["u"] for e in linear_cache]
        x_list = [e["x"] for e in linear_cache]
        lin_modules = [e["module"] for e in linear_cache]

        # h_i post-activation for i < N-1 ; last uses u directly
        h_list: List[torch.Tensor] = []
        for i in range(N):
            h_list.append(self._act(u_list[i]) if i < N - 1 else u_list[i])

        cond = self.condition_on_label and allow_condition_on_label
        onehot = None
        if cond:
            if y is None:
                raise ValueError("cDNI activé mais y=None")
            onehot = self._onehot(y, int(self._num_classes or g_out.size(-1)))

        # backward recursion: boundary grads
        g_list: List[Optional[torch.Tensor]] = [None] * N
        g_list[N - 1] = g_out

        delta_u: List[Optional[torch.Tensor]] = [None] * N
        delta_u[N - 1] = g_out

        sg_total_loss = torch.tensor(0.0, device=device)

        for i in range(N - 2, -1, -1):
            j = i + 1

            # delta_u_j = dL/du_j
            if j < N - 1:
                delta_u_j = g_list[j] * self._act_grad(u_list[j])
            else:
                delta_u_j = g_list[j]
            delta_u[j] = delta_u_j

            # target_i = dL/dh_i = delta_u_j @ W_j
            Wj = lin_modules[j].weight.detach()
            target = delta_u_j @ Wj

            # SG prediction + BP(lambda)
            if key in self._sg_models and len(self._sg_models[key]) > 0:
                hi = h_list[i].detach()
                inp = torch.cat([hi, onehot], dim=1) if (cond and onehot is not None) else hi
                sg_pred = self._sg_models[key][i](inp)

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

        # write grads for linears
        B = float(x_list[0].size(0))
        for i in range(N):
            lin: nn.Linear = lin_modules[i]
            du = delta_u[i]
            if du is None:
                continue
            lin.weight.grad = (du.T @ x_list[i]) / B
            if lin.bias is not None:
                lin.bias.grad = du.mean(dim=0)

        return sg_total_loss

    # ---------------- main API ----------------

    def train_step(self, model: nn.Module, task: Any, batch, device: str, state=None) -> dict:
        model.train()

        if isinstance(batch, Mapping):
            raise NotImplementedError("DNI: Mapping/HF batches not supported in this minimal version.")

        x, y = to_device(batch, device)
        is_rl = isinstance(y, Mapping)

        # 1) forward (no autograd) + capture Linear x/u
        # RL actor-critic: capture actor + critic separately if possible
        with torch.no_grad():
            out = model(x)

        stats: Dict[str, float] = {}

        # 2) build SG models lazily + optimizers
        self._ensure_optimizers(model)

        # 3) compute anchor grads on outputs only
        #    then run DNI chain(s) to write grads, then optimizer.step
        assert self._net_optimizer is not None
        self._net_optimizer.zero_grad(set_to_none=True)

        sg_total_loss = torch.tensor(0.0, device=device)

        # ---------------- RL ----------------
        if is_rl:
            # Actor-Critic dict output
            if isinstance(out, Mapping) and ("logits" in out) and ("value" in out):
                if not (hasattr(model, "actor") and hasattr(model, "critic")):
                    raise RuntimeError(
                        "DNI RL actor-critic: j'attends model.actor et model.critic (comme ActorCriticDiscrete)."
                    )

                # capture actor chain
                actor_cache, actor_hooks = self._capture_linear_xu_hooks(model.actor)
                critic_cache, critic_hooks = self._capture_linear_xu_hooks(model.critic)

                # redo forward once to fill caches (toujours no_grad)
                with torch.no_grad():
                    out2 = model(x)
                for h in actor_hooks + critic_hooks:
                    h.remove()

                logits = out2["logits"]
                value = out2["value"]  # [B] typiquement

                # autograd only on outputs
                logits_var = logits.detach().requires_grad_(True)
                value_var = value.detach().requires_grad_(True)

                loss_out = task.loss({"logits": logits_var, "value": value_var}, y)
                loss, extra = _parse_loss(loss_out)
                stats["loss"] = float(loss.detach().item())
                stats.update(extra)

                g_logits, g_value = torch.autograd.grad(loss, (logits_var, value_var))

                # build SG models (no label conditioning in RL)
                self._maybe_build_sg_models("actor", task, torch.tensor(0, device=device), actor_cache, logits, device, allow_condition_on_label=False)
                self._maybe_build_sg_models("critic", task, torch.tensor(0, device=device), critic_cache, value_var.unsqueeze(-1), device, allow_condition_on_label=False)

                # critic grad shape -> [B,1]
                g_value_2d = g_value.detach().unsqueeze(-1) if g_value.dim() == 1 else g_value.detach()

                sg_total_loss = sg_total_loss + self._dni_chain("actor", actor_cache, g_logits.detach(), None, False, device)
                sg_total_loss = sg_total_loss + self._dni_chain("critic", critic_cache, g_value_2d, None, False, device)

            # DQN/QNet tensor output
            elif torch.is_tensor(out):
                # capture whole model chain
                linear_cache, hooks = self._capture_linear_xu_hooks(model)
                with torch.no_grad():
                    q = model(x)
                for h in hooks:
                    h.remove()

                q_var = q.detach().requires_grad_(True)
                loss_out = task.loss(q_var, y)
                loss, extra = _parse_loss(loss_out)
                stats["loss"] = float(loss.detach().item())
                stats.update(extra)

                (g_q,) = torch.autograd.grad(loss, (q_var,))

                self._maybe_build_sg_models("main", task, torch.tensor(0, device=device), linear_cache, q, device, allow_condition_on_label=False)
                sg_total_loss = sg_total_loss + self._dni_chain("main", linear_cache, g_q.detach(), None, False, device)

            else:
                raise TypeError("DNI RL: sortie modèle non supportée (dict logits/value ou tensor q).")

        # ---------------- Supervised (identique esprit) ----------------
        else:
            # capture whole model chain
            linear_cache, hooks = self._capture_linear_xu_hooks(model)
            with torch.no_grad():
                out2 = model(x)
            for h in hooks:
                h.remove()

            logits = out2["logits"] if isinstance(out2, Mapping) else out2

            # prefer last Linear's u if it matches logits shape
            if linear_cache and torch.is_tensor(logits) and linear_cache[-1]["u"].shape == logits.shape:
                logits = linear_cache[-1]["u"]

            # build SG models lazily
            self._maybe_build_sg_models("main", task, y, linear_cache, logits, device, allow_condition_on_label=True)
            self._ensure_optimizers(model)

            # anchor grad at output only
            logits_var = logits.detach().requires_grad_(True)
            loss_out = task.loss(logits_var, y)
            loss, extra = _parse_loss(loss_out)
            (g_out,) = torch.autograd.grad(loss, (logits_var,))
            stats["loss"] = float(loss.detach().item())
            stats.update(extra)

            if hasattr(task, "metrics"):
                try:
                    stats.update(task.metrics(logits, y))
                except Exception:
                    pass

            sg_total_loss = sg_total_loss + self._dni_chain(
                "main", linear_cache, g_out.detach(), y, True, device
            )

        # 4) apply updates to main net (optimizer step)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        self._net_optimizer.step()

        # 5) update SG models
        if self._sg_optimizer is None:
            self._ensure_optimizers(model)

        if self._sg_optimizer is not None and len(list(self._sg_models.parameters())) > 0:
            self._sg_optimizer.zero_grad(set_to_none=True)
            sg_total_loss.backward()
            self._sg_optimizer.step()
            stats["sg_loss"] = float(sg_total_loss.detach().item())

        self.global_step += 1
        return stats
