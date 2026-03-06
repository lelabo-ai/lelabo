from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OptimizerUpdateRule
from ...core.batch import extract_loss_and_stats, to_device
from ...models.cache_provider import (
    KNOWN_ACTIVATION_MODULE_TYPES,
    CacheSpec,
    forward_with_standard_cache,
)


def _dedup_types(items: Sequence[type[nn.Module]]) -> tuple[type[nn.Module], ...]:
    seen: set[type[nn.Module]] = set()
    out: list[type[nn.Module]] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


class SoftContrastiveLearning(OptimizerUpdateRule):
    """
    SCL (legacy-style) on top of cache v3:

    - Prefer `views["model_blocks"]` when available, else fallback to `views["ordered_blocks"]`.
    - Hidden blocks are updated one by one with local SupCon (autograd).
    - Output block is updated with the runner optimizer using supervised loss.
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
        depth_lr_gamma: float = 1.0,
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

    def _cache_spec_for_model(self, model: nn.Module) -> CacheSpec:
        blocks = self._model_blocks(model)
        if blocks:
            observed_names = tuple(str(getattr(b, "name")) for b in blocks)
            module_types = _dedup_types(
                [
                    type(getattr(b, "module"))
                    for b in blocks
                    if isinstance(getattr(b, "module", None), nn.Module)
                ]
            )
            if not module_types:
                module_types = (nn.Linear, nn.Conv2d)
            return CacheSpec(
                trainable_module_types=module_types,
                observed_module_types=module_types,
                observed_module_names=observed_names,
                capture_inputs=True,
                capture_outputs=True,
                capture_all_calls=True,
                capture_steps=True,
                require_single_call=True,
                require_single_output_head=True,
                include_model_blocks=True,
                include_local_blocks=False,
                auto_pair_post_activation=False,
            )

        trainable_types = (nn.Linear, nn.Conv2d)
        return CacheSpec(
            trainable_module_types=trainable_types,
            capture_inputs=True,
            capture_outputs=True,
            capture_all_calls=True,
            capture_steps=True,
            require_single_call=True,
            require_single_output_head=True,
            include_model_blocks=False,
            include_local_blocks=False,
            auto_pair_post_activation=False,
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
    def _set_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = bool(flag)

    @staticmethod
    def _supervised_contrastive_loss(
        z: torch.Tensor,
        labels: torch.Tensor,
        *,
        tau: float,
        eps: float,
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
            if rep in ("mean", "avg", "gap", "pool"):
                return h.mean(dim=1)
            if rep in ("last", "eos"):
                return h[:, -1, :]
            return h[:, 0, :]
        if h.dim() == 4:
            return h.flatten(1)
        return h

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
                "Set update_rule.params.local_optim to 'sgd' or 'adamw'."
            ) from exc

    @staticmethod
    def _proj_key(name: str) -> str:
        return str(name).replace(".", "__")

    def _make_projection(self, in_dim: int, device: torch.device) -> nn.Module:
        out_dim = min(int(self.proj_dim), int(in_dim)) if int(self.proj_dim) > 0 else int(in_dim)
        hidden = int(self.proj_hidden_dim) if self.proj_hidden_dim is not None else int(out_dim)
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
        if not params:
            raise RuntimeError(f"SCL block '{name}' has no parameters for a local optimizer.")
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
            opt = torch.optim.SGD(params, lr=lr_here, momentum=0.9, weight_decay=wd)
        elif self.local_optim in {"adamw", "adam"}:
            wd = (
                float(self.local_weight_decay)
                if self.local_weight_decay is not None
                else float(self.optimizer.param_groups[0].get("weight_decay", 0.0))
            )
            if self.local_optim == "adam":
                opt = torch.optim.Adam(params, lr=lr_here, weight_decay=wd)
            else:
                opt = torch.optim.AdamW(params, lr=lr_here, weight_decay=wd)
        else:
            raise ValueError(
                f"Unsupported SCL local_optim '{self.local_optim}'. "
                "Expected one of: None/same/runner/copy/sgd/adam/adamw."
            )

        self._local_opt_by_name[name] = opt
        return proj, opt

    @staticmethod
    def _as_view_blocks(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            module = item.get("module")
            if not isinstance(module, nn.Module):
                continue
            out.append(
                {
                    "name": str(item.get("name", "")),
                    "module": module,
                    "rep": str(item.get("rep", "identity")),
                    "is_output": bool(item.get("is_output", False)),
                }
            )
        return out

    def _ordered_training_blocks(self, views: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
        model_blocks = self._as_view_blocks(views.get("model_blocks", []))
        ordered_blocks = self._as_view_blocks(views.get("ordered_blocks", []))
        blocks = model_blocks if model_blocks else ordered_blocks
        if not blocks:
            raise RuntimeError("SCL requires non-empty model blocks in views.")

        output_idx = -1
        for i, block in enumerate(blocks):
            if bool(block.get("is_output", False)):
                output_idx = i
                break
        if output_idx < 0:
            output_idx = len(blocks) - 1
            blocks[output_idx]["is_output"] = True
        return blocks, output_idx

    @staticmethod
    def _initial_input_from_mapping(batch: Mapping[str, Any]) -> torch.Tensor:
        preferred = ("input_ids", "pixel_values", "inputs_embeds", "input_values", "x", "inputs")
        for key in preferred:
            value = batch.get(key, None)
            if torch.is_tensor(value):
                return value
        ignored = {"labels", "label", "attention_mask", "token_type_ids", "position_ids"}
        tensors = [v for k, v in batch.items() if (k not in ignored) and torch.is_tensor(v)]
        if len(tensors) == 1:
            return tensors[0]
        raise RuntimeError("SCL mapping batch: cannot infer the first block input tensor.")

    @staticmethod
    def _adapt_for_linear_input(
        h: torch.Tensor,
        *,
        in_features: int,
        rep: str,
    ) -> torch.Tensor:
        if h.dim() == 2:
            if int(h.size(1)) != int(in_features):
                raise RuntimeError(
                    f"SCL linear input mismatch: got {tuple(h.shape)}, expected in_features={int(in_features)}."
                )
            return h

        rep_l = str(rep).lower()
        candidates: list[torch.Tensor] = []
        if h.dim() == 3:
            if rep_l in ("mean", "avg", "gap", "pool"):
                candidates.append(h.mean(dim=1))
            elif rep_l in ("last", "eos"):
                candidates.append(h[:, -1, :])
            else:
                candidates.append(h[:, 0, :])
            candidates.append(h.reshape(h.size(0), -1))
        elif h.dim() == 4:
            candidates.append(h.flatten(1))
            candidates.append(h.mean(dim=(2, 3)))
        else:
            candidates.append(h.reshape(h.size(0), -1))

        for cand in candidates:
            if cand.dim() == 2 and int(cand.size(1)) == int(in_features):
                return cand

        shapes = ", ".join(str(tuple(c.shape)) for c in candidates)
        raise RuntimeError(
            f"SCL cannot adapt tensor shape {tuple(h.shape)} to Linear(in_features={int(in_features)}). "
            f"Tried: {shapes}."
        )

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

    def _run_block(
        self,
        module: nn.Module,
        xin: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if xin.dim() == 3 and attention_mask is not None:
            for kw in ("attention_mask", "src_key_padding_mask"):
                try:
                    out = module(xin, **{kw: attention_mask})
                    return self._unpack_block_output(out)
                except TypeError:
                    continue
        out = module(xin)
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
            x_curr = self._initial_input_from_mapping(b)
            _out, _cache, views = forward_with_standard_cache(
                model,
                cache_spec=self._cache_spec_for_model(model),
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
            x_curr = x
            _out, _cache, views = forward_with_standard_cache(
                model,
                x,
                cache_spec=self._cache_spec_for_model(model),
            )

        blocks, output_idx = self._ordered_training_blocks(views)
        hidden_blocks = blocks[:output_idx]
        output_block = blocks[output_idx]
        output_module = output_block["module"]

        local_losses: list[float] = []
        depth = 0
        for block in hidden_blocks:
            name = str(block.get("name", ""))
            module = block["module"]
            rep = str(block.get("rep", "identity"))

            x_in = x_curr
            if isinstance(module, nn.Linear):
                x_in = self._adapt_for_linear_input(x_curr, in_features=int(module.in_features), rep=rep)

            params = [p for p in module.parameters() if isinstance(p, nn.Parameter)]
            if params:
                h_local = self._run_block(
                    module,
                    x_in.detach(),
                    attention_mask=attention_mask if torch.is_tensor(attention_mask) else None,
                )
                v_local = self._select_representation(h_local, rep=rep)
                if v_local.dim() != 2:
                    raise RuntimeError(
                        f"SCL local representation for block '{name}' must be 2D, got {tuple(v_local.shape)}."
                    )

                proj, opt = self._ensure_local_supcon_modules(
                    name=name,
                    module=module,
                    device=torch.device(device),
                    depth=depth,
                    feat_dim_override=int(v_local.size(1)),
                )
                self._set_requires_grad(proj, False)
                opt.zero_grad(set_to_none=True)
                z = proj(v_local)
                local_loss = self._supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)
                local_loss.backward()
                opt.step()
                local_losses.append(float(local_loss.detach().item()))
                depth += 1

            with torch.no_grad():
                x_curr = self._run_block(
                    module,
                    x_in,
                    attention_mask=attention_mask if torch.is_tensor(attention_mask) else None,
                )

        if not isinstance(output_module, nn.Module):
            raise RuntimeError("SCL output block module is invalid.")
        head_params = [p for p in output_module.parameters() if isinstance(p, nn.Parameter)]
        head_ids = {id(p) for p in head_params}
        if not head_params:
            raise RuntimeError("SCL output head exposes no trainable parameters.")

        if isinstance(output_module, nn.Linear):
            x_head = self._adapt_for_linear_input(
                x_curr,
                in_features=int(output_module.in_features),
                rep=str(output_block.get("rep", "identity")),
            )
            out = output_module(x_head)
        else:
            out = self._run_block(
                output_module,
                x_curr,
                attention_mask=attention_mask if torch.is_tensor(attention_mask) else None,
            )

        self.zero_grad()
        loss, stats = self._best_effort_stats(task, out, y)
        loss.backward()
        self._clear_non_head_grads(model, head_ids)
        self.step(head_params, require_grads=True, check_finite_grads=True)

        out_stats = dict(stats)
        out_stats["loss"] = float(loss.detach().item())
        out_stats["supcon_loss"] = float(sum(local_losses) / len(local_losses)) if local_losses else 0.0
        self._mark_step_done()
        return out_stats
