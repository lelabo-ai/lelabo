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
    Legacy-like SCL with block-level local updates using autograd.

    - Output block (is_output=True): supervised update via runner optimizer.
    - Hidden blocks: local SupCon per block, with frozen projection heads.
      * 2D input: linear-like local update
      * 3D input: transformer-like local update (rep selection)
      * 4D input: feature-like local update (flatten)
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
        eps: float = 1e-8,
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
        self.eps = float(eps)

        self._proj_by_name = nn.ModuleDict()
        self._local_opt_by_name: dict[str, torch.optim.Optimizer] = {}
        self._local_param_ids_by_name: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _model_blocks(model: nn.Module) -> list[Any]:
        if not hasattr(model, "get_blocks"):
            return []
        try:
            raw = model.get_blocks()
        except Exception:
            return []
        if not isinstance(raw, list):
            return []

        out: list[Any] = []
        for item in raw:
            name = getattr(item, "name", None)
            module = getattr(item, "module", None)
            if isinstance(name, str) and isinstance(module, nn.Module):
                out.append(item)
        return out

    def _cache_spec_for_model(self, model: nn.Module, *, local_block_mode: str = "full_blocks") -> CacheSpec:
        blocks = self._model_blocks(model)
        if blocks:
            observed_names = tuple(str(getattr(b, "name")) for b in blocks)
            module_types = tuple(
                dict.fromkeys(
                    type(getattr(b, "module"))
                    for b in blocks
                    if isinstance(getattr(b, "module", None), nn.Module)
                )
            )
            if not module_types:
                module_types = (nn.Linear, nn.Conv2d)
            return CacheSpec(
                observed_module_names=observed_names,
                observed_module_types=module_types,
                param_module_types=module_types,
                require_block_inputs=True,
                require_single_call=True,
                require_single_output_head=True,
                local_block_mode=str(local_block_mode),
            )

        return CacheSpec(
            param_module_types=(nn.Linear, nn.Conv2d),
            require_block_inputs=True,
            require_single_call=True,
            require_single_output_head=True,
            local_block_mode=str(local_block_mode),
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
    def _feature_dim_for_module(module: nn.Module) -> int | None:
        if isinstance(module, nn.Linear):
            return int(module.out_features)
        if isinstance(module, nn.Conv2d):
            return int(module.out_channels)

        conv = getattr(module, "conv", None)
        if isinstance(conv, nn.Conv2d):
            return int(conv.out_channels)

        conv3 = getattr(module, "conv3", None)
        if isinstance(conv3, nn.Conv2d):
            return int(conv3.out_channels)
        conv2 = getattr(module, "conv2", None)
        if isinstance(conv2, nn.Conv2d):
            return int(conv2.out_channels)

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
        feat_dim_override: int | None = None,
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        feat_dim = int(feat_dim_override) if feat_dim_override is not None else self._feature_dim_for_module(module)
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
    def _unpack_block_output(out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            return out.last_hidden_state
        raise RuntimeError(f"Unsupported block output type: {type(out)}")

    @staticmethod
    def _select_representation(h: torch.Tensor, rep: str) -> torch.Tensor:
        rep = str(rep).lower()
        if h.dim() == 3:
            if rep in ("cls", "class", "first"):
                return h[:, 0, :]
            if rep in ("mean", "avg", "gap", "pool"):
                return h.mean(dim=1)
            if rep in ("last", "eos"):
                return h[:, -1, :]
            return h[:, 0, :]
        if h.dim() == 4:
            return h.flatten(1)
        return h

    @staticmethod
    def _build_extended_attention_mask(model: nn.Module, batch: Mapping[str, Any], device: torch.device):
        attn = batch.get("attention_mask", None)
        if attn is None or (not torch.is_tensor(attn)):
            return None
        input_shape = batch.get("input_ids", attn)
        if torch.is_tensor(input_shape):
            input_shape = input_shape.shape

        if hasattr(model, "get_extended_attention_mask"):
            try:
                return model.get_extended_attention_mask(attn, input_shape, device=device)
            except TypeError:
                try:
                    return model.get_extended_attention_mask(attn, input_shape)
                except Exception:
                    pass

        base = getattr(model, "bert", None)
        if base is not None and hasattr(base, "get_extended_attention_mask"):
            try:
                return base.get_extended_attention_mask(attn, input_shape, device=device)
            except TypeError:
                try:
                    return base.get_extended_attention_mask(attn, input_shape)
                except Exception:
                    pass

        return None

    def _call_transformer_block(self, block: nn.Module, h_in: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is not None:
            for kw in ("attention_mask", "src_key_padding_mask"):
                try:
                    out = block(h_in, **{kw: attention_mask})
                    return self._unpack_block_output(out)
                except TypeError:
                    continue
        out = block(h_in)
        return self._unpack_block_output(out)

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
        stats: dict[str, float] = {}

        loss = None
        if hasattr(out, "loss") and out.loss is not None and torch.is_tensor(out.loss):
            loss = out.loss
        else:
            logits = SoftContrastiveLearning._tensor_output(out)
            loss_res = task.loss(logits, y)
            loss, extra = extract_loss_and_stats(loss_res)
            stats.update({str(k): float(v) for k, v in extra.items()})

        if hasattr(task, "metrics"):
            for candidate in (out, SoftContrastiveLearning._tensor_output(out)):
                try:
                    met = task.metrics(candidate, y)
                    if isinstance(met, Mapping):
                        for k, v in met.items():
                            if isinstance(v, (int, float)):
                                stats[str(k)] = float(v)
                    break
                except Exception:
                    continue

        if loss is None:
            raise RuntimeError("SCL could not compute a valid loss tensor.")
        return loss, stats

    @torch.no_grad()
    def _clear_non_head_grads(self, model: nn.Module, head_ids: set[int]) -> None:
        for p in model.parameters():
            if id(p) in head_ids:
                continue
            if p.grad is not None:
                p.grad = None

    def _local_supcon_update_vector_block(
        self,
        block: nn.Module,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        self._set_requires_grad(proj, False)
        opt.zero_grad(set_to_none=True)

        h = block(xin.detach())
        h_t = self._unpack_block_output(h)
        if h_t.dim() != 2:
            raise RuntimeError(f"SCL vector block must output 2D tensor, got {tuple(h_t.shape)}")

        z = proj(h_t)
        loss = self._supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)
        loss.backward()
        opt.step()
        return float(loss.detach().item())

    def _local_supcon_update_feature_block(
        self,
        block: nn.Module,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        self._set_requires_grad(proj, False)
        opt.zero_grad(set_to_none=True)

        h = block(xin.detach())
        h_t = self._unpack_block_output(h)
        if h_t.dim() != 4:
            raise RuntimeError(f"Feature block must output 4D tensor, got {tuple(h_t.shape)}")

        v = h_t.flatten(1)
        z = proj(v)
        loss = self._supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)
        loss.backward()
        opt.step()
        return float(loss.detach().item())

    def _local_supcon_update_transformer_block(
        self,
        block: nn.Module,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,
        labels: torch.Tensor,
        *,
        rep: str,
        attention_mask: torch.Tensor | None,
    ) -> float:
        self._set_requires_grad(proj, False)
        opt.zero_grad(set_to_none=True)

        h = self._call_transformer_block(block, xin.detach(), attention_mask=attention_mask)
        v = self._select_representation(h, rep=rep)
        z = proj(v)
        loss = self._supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)
        loss.backward()
        opt.step()
        return float(loss.detach().item())

    def train_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        model.train()

        attention_mask = None
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y = b.get("labels", None)
            if not torch.is_tensor(y):
                raise RuntimeError("SCL mapping batches must contain tensor key 'labels'.")
            labels = self._label_indices(y)
            ext_attention_mask = self._build_extended_attention_mask(model, b, device=torch.device(device))
            attention_mask = ext_attention_mask if torch.is_tensor(ext_attention_mask) else b.get("attention_mask", None)
            use_hf_mode = True
            out, _cache, views = forward_with_standard_cache(
                model,
                cache_spec=self._cache_spec_for_model(
                    model,
                    local_block_mode="hf_hidden_states_full_blocks" if use_hf_mode else "full_blocks",
                ),
                **b,
            )
        else:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise RuntimeError(f"SCL expects batch=(x, y) or mapping batch, got {type(batch)}.")
            x, y = batch
            x = to_device(x, device)
            y = to_device(y, device)
            if not torch.is_tensor(y):
                raise RuntimeError(f"SCL expects tensor labels, got {type(y)}.")
            labels = self._label_indices(y)
            out, _cache, views = forward_with_standard_cache(
                model,
                x,
                cache_spec=self._cache_spec_for_model(model, local_block_mode="reconstruct_local_blocks"),
            )

        output_blocks = views.get("output_blocks", [])
        local_blocks = views.get("local_blocks", [])
        if not isinstance(output_blocks, list) or len(output_blocks) != 1:
            raise RuntimeError("SCL expects exactly one output block.")
        if not isinstance(local_blocks, list):
            raise RuntimeError("SCL expects views['local_blocks'] list.")

        output_block = output_blocks[0]
        head_params = [p for p in output_block.iter_params() if isinstance(p, nn.Parameter)]
        head_ids = {id(p) for p in head_params}
        if not head_params:
            raise RuntimeError("SCL output head exposes no parameters to optimize.")

        self.zero_grad()
        loss, stats = self._best_effort_stats(task, out, y)
        loss.backward()
        self._clear_non_head_grads(model, head_ids)
        self.step(head_params, require_grads=True, check_finite_grads=True)

        local_losses: list[float] = []
        depth = 0
        for local in local_blocks:
            if bool(local.get("is_output", False)):
                continue

            name = str(local.get("name", ""))
            module = local.get("module")
            if not isinstance(module, nn.Module):
                continue

            xin = local.get("x")
            if not torch.is_tensor(xin):
                continue

            if xin.dim() == 2:
                feat_dim_override = None
                u_cache = local.get("u")
                if torch.is_tensor(u_cache):
                    if int(u_cache.size(0)) <= 0:
                        raise RuntimeError(f"SCL local block '{name}' has empty cached output batch.")
                    if u_cache.dim() == 3:
                        feat_dim_override = int(u_cache.size(-1))
                        proj, opt = self._ensure_local_supcon_modules(
                            name=name,
                            module=module,
                            device=torch.device(device),
                            depth=depth,
                            feat_dim_override=feat_dim_override,
                        )
                        rep = str(local.get("rep", "cls"))
                        local_losses.append(
                            self._local_supcon_update_transformer_block(
                                module,
                                proj,
                                opt,
                                xin,
                                labels,
                                rep=rep,
                                attention_mask=attention_mask if torch.is_tensor(attention_mask) else None,
                            )
                        )
                        depth += 1
                        continue
                    feat_dim_override = int(u_cache.reshape(u_cache.size(0), -1).size(1))
                proj, opt = self._ensure_local_supcon_modules(
                    name=name,
                    module=module,
                    device=torch.device(device),
                    depth=depth,
                    feat_dim_override=feat_dim_override,
                )
                local_losses.append(self._local_supcon_update_vector_block(module, proj, opt, xin, labels))
                depth += 1
                continue

            if xin.dim() == 3:
                proj, opt = self._ensure_local_supcon_modules(
                    name=name,
                    module=module,
                    device=torch.device(device),
                    depth=depth,
                    feat_dim_override=int(xin.size(-1)),
                )
                rep = str(local.get("rep", "cls"))
                local_losses.append(
                    self._local_supcon_update_transformer_block(
                        module,
                        proj,
                        opt,
                        xin,
                        labels,
                        rep=rep,
                        attention_mask=attention_mask if torch.is_tensor(attention_mask) else None,
                    )
                )
                depth += 1
                continue

            if xin.dim() == 4:
                feat_dim_override = None
                u_cache = local.get("u")
                if torch.is_tensor(u_cache):
                    if int(u_cache.size(0)) <= 0:
                        raise RuntimeError(f"SCL local block '{name}' has empty cached output batch.")
                    feat_dim_override = int(u_cache.reshape(u_cache.size(0), -1).size(1))
                if feat_dim_override is None:
                    raise RuntimeError(
                        "SCL feature-local update with flatten requires cached tensor output "
                        f"for block '{name}' to infer projection input dim."
                    )
                proj, opt = self._ensure_local_supcon_modules(
                    name=name,
                    module=module,
                    device=torch.device(device),
                    depth=depth,
                    feat_dim_override=feat_dim_override,
                )
                local_losses.append(self._local_supcon_update_feature_block(module, proj, opt, xin, labels))
                depth += 1
                continue

        out_stats = dict(stats)
        out_stats["loss"] = float(loss.detach().item())
        out_stats["supcon_loss"] = float(sum(local_losses) / len(local_losses)) if local_losses else 0.0
        self._mark_step_done()
        return out_stats
