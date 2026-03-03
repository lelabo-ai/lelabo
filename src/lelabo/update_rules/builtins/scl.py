from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OptimizerUpdateRule
from ...core.batch import extract_loss_and_stats, to_device
from ...models.cache_provider import CacheSpec, forward_with_standard_cache


class SoftContrastiveLearning(OptimizerUpdateRule):
    """
    Minimal SCL:
    - output head: supervised BP (runner optimizer)
    - hidden Linear/Conv2d blocks: local SupCon updates with frozen random projection
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        supcon_tau: float = 0.1,
        local_lr: float | None = None,
        local_optim: str | None = None,
        local_weight_decay: float | None = None,
        proj_dim: int = 256,
        proj_hidden_dim: int | None = None,
        depth_lr_gamma: float = 1,
        depth_lr_min_factor: float = 0.01,
        depth_lr_max_factor: float = 100.0,
        grad_clip: float | None = None,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            grad_clip=grad_clip,
            strict_require_grads=True,
            check_finite_grads=True,
        )
        self.supcon_tau = float(supcon_tau)
        self.local_lr = None if local_lr is None else float(local_lr)
        self.local_optim = None if local_optim is None else str(local_optim).lower()
        self.local_weight_decay = None if local_weight_decay is None else float(local_weight_decay)
        self.proj_dim = int(proj_dim)
        self.proj_hidden_dim = None if proj_hidden_dim is None else int(proj_hidden_dim)
        self.depth_lr_gamma = float(depth_lr_gamma)
        self.depth_lr_min_factor = float(depth_lr_min_factor)
        self.depth_lr_max_factor = float(depth_lr_max_factor)

        self._proj_by_name = nn.ModuleDict()
        self._local_opt_by_name: dict[str, torch.optim.Optimizer] = {}
        self._local_param_ids_by_name: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _cache_spec() -> CacheSpec:
        return CacheSpec(
            param_module_types=(nn.Linear, nn.Conv2d),
            require_block_inputs=True,
            require_single_call=True,
            require_single_output_head=True,
        )

    @staticmethod
    def _label_indices(y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 2 and int(y.size(1)) == 1 and not torch.is_floating_point(y):
            return y.view(-1).long()
        if y.dim() == 2 and torch.is_floating_point(y):
            return y.argmax(dim=1).long()
        if y.dim() == 1:
            return y.long()
        raise RuntimeError(f"SCL expects class labels [B], [B,1], or one-hot [B,C], got {tuple(y.shape)}.")

    @staticmethod
    def _supervised_contrastive_loss(
        z: torch.Tensor,
        labels: torch.Tensor,
        *,
        tau: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        batch = int(z.size(0))
        z_n = F.normalize(z, dim=1)
        sim = (z_n @ z_n.t()) / float(tau)
        sim = sim - sim.max(dim=1, keepdim=True).values

        labels_col = labels.view(-1, 1)
        pos_mask = (labels_col == labels_col.t()).float()
        pos_mask.fill_diagonal_(0.0)

        eye = torch.eye(batch, device=z.device, dtype=z.dtype)
        exp_sim = torch.exp(sim) * (1.0 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + float(eps))
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + float(eps))
        return -mean_log_prob_pos.mean()

    @staticmethod
    def _feature_dim(module: nn.Module) -> int | None:
        if isinstance(module, nn.Linear):
            return int(module.out_features)
        if isinstance(module, nn.Conv2d):
            return int(module.out_channels)
        return None

    @staticmethod
    def _proj_key(name: str) -> str:
        return str(name).replace(".", "__")

    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = bool(flag)

    def _depth_scaled_lr(self, depth: int) -> float:
        factor = self.depth_lr_gamma ** int(depth)
        factor = max(self.depth_lr_min_factor, min(self.depth_lr_max_factor, factor))
        base_lr = self.local_lr
        if base_lr is None:
            if not self.optimizer.param_groups:
                raise RuntimeError("SCL cannot infer local_lr from an empty runner optimizer.")
            base_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        return float(base_lr * factor)

    @staticmethod
    def _set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
        for group in opt.param_groups:
            group["lr"] = float(lr)

    def _make_runner_local_optimizer(
        self,
        params: list[nn.Parameter],
        *,
        lr: float,
    ) -> torch.optim.Optimizer:
        defaults = dict(getattr(self.optimizer, "defaults", {}))
        sig = inspect.signature(self.optimizer.__class__.__init__)
        allowed = {name for name in sig.parameters.keys() if name not in {"self", "params"}}
        init_kwargs = {k: v for k, v in defaults.items() if k in allowed}
        init_kwargs["lr"] = float(lr)
        if self.local_weight_decay is not None and "weight_decay" in allowed:
            init_kwargs["weight_decay"] = float(self.local_weight_decay)
        try:
            return self.optimizer.__class__(params, **init_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "SCL failed to clone runner optimizer for local updates. "
                "Set update_rule.params.local_optim to 'sgd' or 'adamw' to force a local optimizer type."
            ) from exc

    def _make_projection(self, in_dim: int, device: torch.device) -> nn.Module:
        out_dim = min(int(self.proj_dim), int(in_dim)) if int(self.proj_dim) > 0 else int(in_dim)
        hidden = int(self.proj_hidden_dim) if self.proj_hidden_dim is not None else int(in_dim)
        proj = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        ).to(device)
        self._set_requires_grad(proj, False)
        return proj

    def _ensure_local_supcon_modules(
        self,
        *,
        name: str,
        module: nn.Module,
        device: torch.device,
        depth: int,
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        feat_dim = self._feature_dim(module)
        if feat_dim is None:
            raise TypeError(f"Unsupported module for local SCL update: {type(module)}")
        key = self._proj_key(name)

        if key not in self._proj_by_name:
            self._proj_by_name[key] = self._make_projection(feat_dim, device)
        else:
            self._proj_by_name[key] = self._proj_by_name[key].to(device)
            self._set_requires_grad(self._proj_by_name[key], False)
        proj = self._proj_by_name[key]

        params = [p for p in module.parameters() if isinstance(p, nn.Parameter)]
        param_ids = tuple(sorted(id(p) for p in params))
        lr_here = self._depth_scaled_lr(depth)

        if name in self._local_opt_by_name and self._local_param_ids_by_name.get(name) == param_ids:
            opt = self._local_opt_by_name[name]
            self._set_optimizer_lr(opt, lr_here)
            return proj, opt

        self._local_param_ids_by_name[name] = param_ids
        if self.local_optim in (None, "", "same", "runner", "copy"):
            opt = self._make_runner_local_optimizer(params, lr=lr_here)
        elif self.local_optim == "sgd":
            wd = (
                float(self.local_weight_decay)
                if self.local_weight_decay is not None
                else float(self.optimizer.param_groups[0].get("weight_decay", 0.0))
            )
            opt = torch.optim.SGD(
                params,
                lr=lr_here,
                momentum=0.9,
                weight_decay=wd,
            )
        else:
            wd = (
                float(self.local_weight_decay)
                if self.local_weight_decay is not None
                else float(self.optimizer.param_groups[0].get("weight_decay", 0.0))
            )
            opt = torch.optim.AdamW(
                params,
                lr=lr_here,
                weight_decay=wd,
            )
        self._local_opt_by_name[name] = opt
        return proj, opt

    @staticmethod
    def _tensor_output(out: Any) -> torch.Tensor:
        if isinstance(out, Mapping):
            if "logits" in out and torch.is_tensor(out["logits"]):
                return out["logits"]
            raise RuntimeError("SCL expects mapping outputs to contain tensor key 'logits'.")
        if hasattr(out, "logits") and torch.is_tensor(out.logits):
            return out.logits
        if not torch.is_tensor(out):
            raise RuntimeError(f"SCL expects tensor outputs (or mapping/logits attr), got {type(out)}.")
        return out

    @staticmethod
    def _best_effort_stats(task, out: Any, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss_res = task.loss(out, y)
        loss, extra = extract_loss_and_stats(loss_res)
        stats: dict[str, float] = {str(k): float(v) for k, v in extra.items()}
        if hasattr(task, "metrics"):
            try:
                met = task.metrics(out, y)
                if isinstance(met, Mapping):
                    for k, v in met.items():
                        if isinstance(v, (int, float)):
                            stats[str(k)] = float(v)
            except Exception:
                pass
        return loss, stats

    @torch.no_grad()
    def _clear_non_head_grads(self, model: nn.Module, head_ids: set[int]) -> None:
        for p in model.parameters():
            if id(p) in head_ids:
                continue
            if p.grad is not None:
                p.grad = None

    def train_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        model.train()
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise RuntimeError(f"SCL v1 expects batch=(x, y), got {type(batch)}.")

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)
        if not torch.is_tensor(y):
            raise RuntimeError(f"SCL v1 expects tensor labels, got {type(y)}.")
        labels = self._label_indices(y)

        out, cache, views = forward_with_standard_cache(model, x, cache_spec=self._cache_spec())
        output_blocks = views.get("output_blocks", [])
        hidden_blocks = views.get("hidden_blocks", [])
        if not isinstance(output_blocks, list) or len(output_blocks) != 1:
            raise RuntimeError("SCL v1 expects exactly one output block.")

        output_block = output_blocks[0]
        head_params = [p for p in output_block.iter_params() if isinstance(p, nn.Parameter)]
        head_ids = {id(p) for p in head_params}
        if not head_params:
            raise RuntimeError("SCL output head exposes no parameters to optimize.")

        # Head supervised update (runner optimizer).
        self.zero_grad()
        logits = self._tensor_output(out)
        loss, stats = self._best_effort_stats(task, logits, y)
        loss.backward()
        self._clear_non_head_grads(model, head_ids)
        self.step(head_params, require_grads=True, check_finite_grads=True)

        # Local SCL updates on hidden blocks.
        block_inputs = cache.get("block_inputs", {})
        if not isinstance(block_inputs, Mapping):
            raise RuntimeError("SCL expects dict cache['block_inputs'].")

        local_losses: list[float] = []
        for depth, block in enumerate(hidden_blocks):
            name = str(getattr(block, "name", ""))
            module = getattr(block, "module", None)
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                raise NotImplementedError(f"SCL v1 supports Linear/Conv2d hidden blocks, got {type(module)} on '{name}'.")

            x_in = block_inputs.get(name)
            if not torch.is_tensor(x_in):
                raise RuntimeError(f"SCL missing tensor cache['block_inputs'][{name!r}]")
            if isinstance(module, nn.Linear) and x_in.dim() != 2:
                raise RuntimeError(f"SCL hidden Linear '{name}' expects 2D input, got {tuple(x_in.shape)}.")
            if isinstance(module, nn.Conv2d) and x_in.dim() != 4:
                raise RuntimeError(f"SCL hidden Conv2d '{name}' expects 4D input, got {tuple(x_in.shape)}.")

            proj, opt = self._ensure_local_supcon_modules(
                name=name,
                module=module,
                device=torch.device(device),
                depth=depth,
            )
            self._set_requires_grad(proj, False)
            opt.zero_grad(set_to_none=True)

            h = module(x_in.detach())
            if isinstance(module, nn.Conv2d):
                if h.dim() != 4:
                    raise RuntimeError(f"SCL hidden Conv2d '{name}' must output 4D tensor.")
                rep = h.mean(dim=(2, 3))
            else:
                if h.dim() != 2:
                    raise RuntimeError(f"SCL hidden Linear '{name}' must output 2D tensor.")
                rep = h
            z = proj(rep)
            local_loss = self._supervised_contrastive_loss(
                z,
                labels,
                tau=self.supcon_tau,
            )
            local_loss.backward()
            opt.step()
            local_losses.append(float(local_loss.detach().item()))

        out_stats = dict(stats)
        out_stats["loss"] = float(loss.detach().item())
        out_stats["supcon_loss"] = (
            float(sum(local_losses) / len(local_losses)) if local_losses else 0.0
        )
        self._mark_step_done()
        return out_stats
