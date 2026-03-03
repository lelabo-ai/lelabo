from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OptimizerUpdateRule
from ...core.batch import extract_loss_and_stats, to_device
from ...models.cache_provider import CacheSpec, forward_with_standard_cache


class _SGModel(nn.Module):
    """Synthetic-gradient model M(h[, c]) -> dL/dh."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0):
        super().__init__()
        if int(hidden_dim) > 0:
            self.net = nn.Sequential(
                nn.Linear(int(in_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(out_dim)),
            )
            last = self.net[-1]
        else:
            self.net = nn.Linear(int(in_dim), int(out_dim))
            last = self.net

        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNI(OptimizerUpdateRule):
    """
    Decoupled Neural Interfaces (supervised, Linear-chain v1).

    v1 scope:
    - batch=(x, y) only (no mapping/RL).
    - local chain made of Linear layers captured via cache v2.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        sg_lr: float = 3e-4,
        sg_optim: str = "adamw",
        sg_weight_decay: float = 0.0,
        sg_hidden: int = 0,
        condition_on_label: bool = False,
        lambda_mix: float = 0.0,
        sg_scale: float = 1.0,
        activation: str = "relu",
        grad_clip: float | None = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            grad_clip=grad_clip,
            strict_require_grads=True,
            check_finite_grads=True,
        )
        self.sg_lr = float(sg_lr)
        self.sg_optim = str(sg_optim).lower()
        self.sg_weight_decay = float(sg_weight_decay)
        self.sg_hidden = int(sg_hidden)
        self.condition_on_label = bool(condition_on_label)
        self.lambda_mix = float(lambda_mix)
        self.sg_scale = float(sg_scale)
        self.activation = str(activation).lower()

        self._sg_models: nn.ModuleDict = nn.ModuleDict()
        self._sg_signatures: dict[str, tuple[int, ...]] = {}
        self._sg_optimizer: torch.optim.Optimizer | None = None
        self._num_classes: int | None = None

    def _cache_spec(self) -> CacheSpec:
        return CacheSpec(
            param_module_types=(nn.Linear,),
            require_block_inputs=True,
            require_block_outputs=True,
            require_single_call=True,
            require_single_output_head=True,
            require_input_ndim=2,
            require_output_ndim=2,
        )

    def _act(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(u)
        if self.activation == "tanh":
            return torch.tanh(u)
        return u

    def _act_grad(self, u: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return (u > 0).to(u.dtype)
        if self.activation == "tanh":
            t = torch.tanh(u)
            return 1.0 - t * t
        return torch.ones_like(u)

    @staticmethod
    def _as_tensor_output(out: Any) -> torch.Tensor:
        if isinstance(out, Mapping):
            if "logits" in out and torch.is_tensor(out["logits"]):
                return out["logits"]
            raise RuntimeError("DNI v1 mapping output must contain tensor key 'logits'.")
        if not torch.is_tensor(out):
            raise RuntimeError(f"DNI v1 expects tensor output (or mapping with logits), got {type(out)}.")
        return out

    @staticmethod
    def _infer_num_classes(task: Any, y: torch.Tensor, logits: torch.Tensor) -> int:
        if hasattr(task, "num_classes"):
            try:
                return int(getattr(task, "num_classes"))
            except Exception:
                pass
        if logits.dim() >= 2:
            return int(logits.size(-1))
        return int(y.max().item()) + 1

    def _condition_tensor(self, y: torch.Tensor, *, num_classes: int, device, dtype) -> torch.Tensor:
        if y.dim() == 2 and int(y.size(1)) == int(num_classes):
            return y.to(device=device, dtype=dtype)
        if y.dim() == 2 and int(y.size(1)) == 1 and not torch.is_floating_point(y):
            y = y.view(-1)
        if y.dim() != 1:
            raise RuntimeError(
                "DNI condition_on_label expects class labels [B], [B,1], or one-hot [B,C]."
            )
        labels = y.long().to(device=device)
        if int(labels.min().item()) < 0 or int(labels.max().item()) >= int(num_classes):
            raise RuntimeError(
                f"DNI labels out of range for num_classes={int(num_classes)}: "
                f"min={int(labels.min().item())}, max={int(labels.max().item())}."
            )
        onehot = torch.zeros(labels.size(0), int(num_classes), device=device, dtype=dtype)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        return onehot

    def _linear_chain(self, cache: Mapping[str, Any], views: Mapping[str, Any]) -> list[dict[str, Any]]:
        param_blocks = views.get("param_blocks", [])
        if not isinstance(param_blocks, list) or not param_blocks:
            raise RuntimeError("DNI found no linear param blocks in cache views.")
        by_name: dict[str, nn.Linear] = {}
        for block in param_blocks:
            name = str(getattr(block, "name", ""))
            mod = getattr(block, "module", None)
            if not isinstance(mod, nn.Linear):
                raise RuntimeError(f"DNI v1 supports only Linear chains, got {type(mod)} on '{name}'.")
            by_name[name] = mod

        order: list[str] = []
        seen: set[str] = set()
        steps = cache.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                name = str(step.get("name", ""))
                if name in by_name and name not in seen:
                    seen.add(name)
                    order.append(name)
        if len(order) != len(by_name):
            for block in param_blocks:
                name = str(getattr(block, "name", ""))
                if name not in seen:
                    order.append(name)
                    seen.add(name)

        block_inputs = cache.get("block_inputs", {})
        block_outputs = cache.get("block_outputs", {})
        if not isinstance(block_inputs, Mapping) or not isinstance(block_outputs, Mapping):
            raise RuntimeError("DNI expects dict cache['block_inputs'] and cache['block_outputs'].")

        chain: list[dict[str, Any]] = []
        for name in order:
            x = block_inputs.get(name)
            u = block_outputs.get(name)
            if not torch.is_tensor(x) or not torch.is_tensor(u):
                raise RuntimeError(f"DNI missing tensor cache entries for block '{name}'.")
            if x.dim() != 2 or u.dim() != 2:
                raise RuntimeError(f"DNI expects 2D Linear cache tensors on '{name}', got {tuple(x.shape)} / {tuple(u.shape)}.")
            chain.append({"name": name, "module": by_name[name], "x": x, "u": u})
        return chain

    def _maybe_build_sg_models(
        self,
        key: str,
        *,
        task: Any,
        y: torch.Tensor,
        chain: list[dict[str, Any]],
        output_tensor: torch.Tensor,
        device,
    ) -> None:
        if len(chain) < 2:
            self._sg_models[key] = nn.ModuleList()
            self._sg_signatures[key] = tuple()
            self._sg_optimizer = None
            return

        signature_items: list[int] = []
        for entry in chain:
            module = entry["module"]
            signature_items.extend([int(module.in_features), int(module.out_features)])
        signature = tuple(signature_items)

        current = self._sg_models[key] if key in self._sg_models else None
        if (
            self._sg_signatures.get(key) == signature
            and isinstance(current, nn.ModuleList)
            and len(current) == (len(chain) - 1)
        ):
            return

        cond_dim = 0
        if self.condition_on_label:
            cond_dim = self._infer_num_classes(task, y, output_tensor)
            self._num_classes = int(cond_dim)

        models = nn.ModuleList()
        for i in range(len(chain) - 1):
            hidden_dim = int(chain[i]["u"].size(1))
            in_dim = hidden_dim + int(cond_dim)
            models.append(_SGModel(in_dim=in_dim, out_dim=hidden_dim, hidden_dim=self.sg_hidden))

        self._sg_models[key] = models.to(device)
        self._sg_signatures[key] = signature
        self._sg_optimizer = None

    def _ensure_sg_optimizer(self) -> None:
        params = list(self._sg_models.parameters())
        if not params:
            self._sg_optimizer = None
            return
        if self._sg_optimizer is not None:
            return
        if self.sg_optim == "sgd":
            self._sg_optimizer = torch.optim.SGD(
                params,
                lr=self.sg_lr,
                momentum=0.0,
                weight_decay=self.sg_weight_decay,
            )
        else:
            self._sg_optimizer = torch.optim.AdamW(
                params,
                lr=self.sg_lr,
                weight_decay=self.sg_weight_decay,
            )

    def _dni_chain(
        self,
        key: str,
        *,
        chain: list[dict[str, Any]],
        g_out: torch.Tensor,
        cond: torch.Tensor | None,
        device,
    ) -> torch.Tensor:
        if not chain:
            return torch.tensor(0.0, device=device)

        n = len(chain)
        u_list = [entry["u"] for entry in chain]
        x_list = [entry["x"] for entry in chain]
        modules = [entry["module"] for entry in chain]
        h_list: list[torch.Tensor] = [self._act(u_list[i]) if i < n - 1 else u_list[i] for i in range(n)]

        g_list: list[torch.Tensor | None] = [None] * n
        g_list[n - 1] = g_out
        delta_u: list[torch.Tensor | None] = [None] * n
        delta_u[n - 1] = g_out
        sg_total_loss = torch.tensor(0.0, device=device)

        sg_models = self._sg_models[key] if key in self._sg_models else None
        has_sg = isinstance(sg_models, nn.ModuleList) and len(sg_models) > 0

        for i in range(n - 2, -1, -1):
            j = i + 1
            g_j = g_list[j]
            if g_j is None:
                raise RuntimeError("DNI internal error: missing boundary gradient.")
            if j < n - 1:
                delta_u_j = g_j * self._act_grad(u_list[j])
            else:
                delta_u_j = g_j
            delta_u[j] = delta_u_j

            w_j = modules[j].weight.detach()
            target = delta_u_j @ w_j

            if has_sg:
                h_i = h_list[i].detach()
                if cond is not None:
                    inp = torch.cat([h_i, cond], dim=1)
                else:
                    inp = h_i
                sg_pred = sg_models[i](inp)
                sg_total_loss = sg_total_loss + F.mse_loss(sg_pred, target.detach())
                g_i = (self.lambda_mix * target) + ((1.0 - self.lambda_mix) * (self.sg_scale * sg_pred.detach()))
            else:
                g_i = target
            g_list[i] = g_i

        if n > 1:
            g0 = g_list[0]
            if g0 is None:
                raise RuntimeError("DNI internal error: missing gradient at first layer.")
            delta_u[0] = g0 * self._act_grad(u_list[0])
        else:
            delta_u[0] = g_list[0]

        batch_size = float(max(1, int(x_list[0].size(0))))
        for i in range(n):
            du = delta_u[i]
            if du is None:
                continue
            lin = modules[i]
            lin.weight.grad = (du.transpose(0, 1) @ x_list[i]) / batch_size
            if lin.bias is not None:
                lin.bias.grad = du.mean(dim=0)
        return sg_total_loss

    @staticmethod
    def _best_effort_metrics(task, output: torch.Tensor, y: torch.Tensor, stats: dict[str, float]) -> None:
        if not hasattr(task, "metrics"):
            return
        try:
            met = task.metrics(output, y)
        except Exception:
            return
        if isinstance(met, Mapping):
            for key, value in met.items():
                if isinstance(value, (int, float)):
                    stats[str(key)] = float(value)

    def train_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        model.train()
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise RuntimeError(f"DNI v1 expects batch=(x, y), got {type(batch)}.")

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)
        if isinstance(y, Mapping):
            raise NotImplementedError("DNI v1 does not support RL/mapping labels.")
        if not torch.is_tensor(y):
            raise RuntimeError(f"DNI v1 expects tensor labels, got {type(y)}.")

        out, cache, views = forward_with_standard_cache(model, x, cache_spec=self._cache_spec())
        chain = self._linear_chain(cache, views)
        out_tensor = self._as_tensor_output(out)
        if out_tensor.dim() != 2:
            raise RuntimeError(f"DNI v1 expects a 2D output tensor, got shape={tuple(out_tensor.shape)}.")

        # Prefer the final Linear pre-activation when shapes align, matching legacy behavior.
        if chain and tuple(chain[-1]["u"].shape) == tuple(out_tensor.shape):
            output_for_loss = chain[-1]["u"]
        else:
            output_for_loss = out_tensor

        self._maybe_build_sg_models(
            "main",
            task=task,
            y=y,
            chain=chain,
            output_tensor=output_for_loss,
            device=output_for_loss.device,
        )
        self._ensure_sg_optimizer()

        logits_var = output_for_loss.detach().requires_grad_(True)
        loss_res = task.loss(logits_var, y)
        loss, extra = extract_loss_and_stats(loss_res)
        (g_out,) = torch.autograd.grad(loss, (logits_var,))

        stats: dict[str, float] = {"loss": float(loss.detach().item())}
        for key, value in extra.items():
            stats[str(key)] = float(value)
        self._best_effort_metrics(task, output_for_loss.detach(), y, stats)

        cond = None
        if self.condition_on_label:
            num_classes = int(self._num_classes or self._infer_num_classes(task, y, output_for_loss))
            cond = self._condition_tensor(
                y,
                num_classes=num_classes,
                device=output_for_loss.device,
                dtype=output_for_loss.dtype,
            )

        self.zero_grad()
        sg_total_loss = self._dni_chain(
            "main",
            chain=chain,
            g_out=g_out.detach(),
            cond=cond,
            device=output_for_loss.device,
        )
        self.step(model.parameters(), require_grads=True, check_finite_grads=True)

        if self._sg_optimizer is not None and sg_total_loss.requires_grad:
            self._sg_optimizer.zero_grad(set_to_none=True)
            sg_total_loss.backward()
            self._sg_optimizer.step()
            stats["sg_loss"] = float(sg_total_loss.detach().item())

        self._mark_step_done()
        return stats
