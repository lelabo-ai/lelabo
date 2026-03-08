from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OptimizerUpdateRule
from ...core.batch import extract_loss_and_stats, to_device
from ...models.cache_provider import CacheSpec, forward_with_standard_cache


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
    Faithful SCL implementation on top of cache v3.

    Intended behavior (close to the original implementation):
      - one snapshot forward with cache
      - hidden blocks updated independently using local SupCon
        from cached block inputs
      - output head updated with the runner optimizer using
        supervised loss
      - no sequential hidden-block propagation during local updates
      - 4D feature maps use GAP before projection
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        supcon_tau: float = 0.1,
        local_lr: float | None = None,
        local_optim: str | None = None,
        local_weight_decay: float | None = None,
        proj_dim: int = 128,
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
        self._pending_local_opt_state_by_name: dict[str, dict[str, Any]] = {}

    # ============================================================
    # block / cache discovery
    # ============================================================

    @staticmethod
    def _declared_block_specs(model: nn.Module) -> list[Any]:
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
        blocks = self._declared_block_specs(model)
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
                declared_blocks_mode="only",
                capture_inputs=True,
                capture_outputs=True,
                capture_all_calls=True,
                capture_steps=True,
                require_single_call=True,
                require_single_output_head=True,
                auto_pair_post_activation=False,
            )

        trainable_types = (nn.Linear, nn.Conv2d)
        return CacheSpec(
            trainable_module_types=trainable_types,
            declared_blocks_mode="ignore",
            capture_inputs=True,
            capture_outputs=True,
            capture_all_calls=True,
            capture_steps=True,
            require_single_call=True,
            require_single_output_head=True,
            auto_pair_post_activation=False,
        )

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
                    "is_trainable": bool(item.get("is_trainable", False)),
                }
            )
        return out

    def _ordered_training_blocks(self, views: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
        declared_blocks = self._as_view_blocks(views.get("declared_blocks", []))
        trainable_segments = self._as_view_blocks(views.get("trainable_segments", []))
        execution_blocks = [
            block
            for block in self._as_view_blocks(views.get("execution_blocks", []))
            if bool(block.get("is_trainable", False))
        ]
        blocks = declared_blocks if declared_blocks else (trainable_segments if trainable_segments else execution_blocks)
        if not blocks:
            raise RuntimeError("SCL requires non-empty declared/trainable execution blocks in views.")

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
    def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                value = mapping[key]
                if value is not None:
                    return value
        return None

    @staticmethod
    def _extract_block_inputs_from_any(raw: Any) -> dict[str, torch.Tensor]:
        """
        Best-effort extractor for a map: block_name -> input tensor.

        Supports several likely cache layouts:
          - raw["module_inputs"][name] = tensor
          - raw["block_inputs"][name] = tensor
          - raw["inputs_by_name"][name] = tensor
          - raw["steps"] / raw["calls"] = list[dict(...)]
        """
        out: dict[str, torch.Tensor] = {}

        if not isinstance(raw, Mapping):
            return out

        for key in ("module_inputs", "block_inputs", "inputs_by_name"):
            value = raw.get(key, None)
            if isinstance(value, Mapping):
                for name, tensor in value.items():
                    if isinstance(name, str) and torch.is_tensor(tensor):
                        out[name] = tensor
                if out:
                    return out

        for seq_key in ("steps", "calls", "records"):
            seq = raw.get(seq_key, None)
            if not isinstance(seq, list):
                continue
            for item in seq:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("name", None)
                if not isinstance(name, str):
                    name = item.get("module_name", None)
                if not isinstance(name, str):
                    continue
                xin = SoftContrastiveLearning._first_present(
                    item,
                    "x",
                    "input",
                    "inputs",
                    "x_in",
                    "xin",
                )
                if torch.is_tensor(xin):
                    out[name] = xin
                elif isinstance(xin, (tuple, list)) and len(xin) > 0 and torch.is_tensor(xin[0]):
                    out[name] = xin[0]
            if out:
                return out

        return out

    def _extract_block_inputs(
        self,
        *,
        cache: Any,
        views: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        # first try cache
        out = self._extract_block_inputs_from_any(cache)
        if out:
            return out

        # then try views
        out = self._extract_block_inputs_from_any(views)
        if out:
            return out

        # finally inspect declared/execution blocks if they embed input tensors
        for key in ("declared_blocks", "execution_blocks"):
            raw = views.get(key, None)
            if not isinstance(raw, list):
                continue
            tmp: dict[str, torch.Tensor] = {}
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("name", None)
                if not isinstance(name, str):
                    continue
                xin = self._first_present(
                    item,
                    "x",
                    "input",
                    "inputs",
                    "x_in",
                    "xin",
                )
                if torch.is_tensor(xin):
                    tmp[name] = xin
                elif isinstance(xin, (tuple, list)) and len(xin) > 0 and torch.is_tensor(xin[0]):
                    tmp[name] = xin[0]
            if tmp:
                return tmp

        raise RuntimeError(
            "SCL could not recover block_inputs from the standard cache. "
            "You likely need to adapt _extract_block_inputs() to your cache_provider layout."
        )

    # ============================================================
    # generic helpers
    # ============================================================

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
            if rep in ("cls", "class", "first"):
                return h[:, 0, :]
            if rep in ("mean", "avg", "gap", "pool"):
                return h.mean(dim=1)
            if rep in ("last", "eos"):
                return h[:, -1, :]
            return h[:, 0, :]
        if h.dim() == 4:
            # faithful to the original implementation: GAP for feature maps
            return h.mean(dim=(2, 3))
        return h

    @staticmethod
    def _feature_dim_for_module(module: nn.Module) -> int | None:
        root = getattr(module, "root", None)
        if isinstance(root, nn.Module) and root is not module:
            return SoftContrastiveLearning._feature_dim_for_module(root)

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

    def _local_supcon_update_dense_block(
        self,
        block: nn.Module,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        was_req = [p.requires_grad for p in block.parameters()]
        self._set_requires_grad(block, True)
        self._set_requires_grad(proj, False)

        h = block(xin)
        if not torch.is_tensor(h) or h.dim() != 2:
            raise RuntimeError(f"Dense block must output 2D tensor, got {type(h)} {getattr(h, 'shape', None)}")

        z = proj(h)
        loss = self._supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for p, prev in zip(block.parameters(), was_req):
            p.requires_grad = prev

        return float(loss.detach().item())

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
                "Set local_optim to 'sgd' or 'adamw'."
            ) from exc

    @staticmethod
    def _proj_key(name: str) -> str:
        return str(name).replace(".", "__")

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

    @staticmethod
    def _state_to_cpu(raw: Any) -> Any:
        if torch.is_tensor(raw):
            return raw.detach().cpu().clone()
        if isinstance(raw, Mapping):
            return {str(k): SoftContrastiveLearning._state_to_cpu(v) for k, v in raw.items()}
        if isinstance(raw, list):
            return [SoftContrastiveLearning._state_to_cpu(v) for v in raw]
        if isinstance(raw, tuple):
            return tuple(SoftContrastiveLearning._state_to_cpu(v) for v in raw)
        return raw

    def _rebuild_proj_modules_from_state(self, state: Mapping[str, Any]) -> None:
        roots = sorted(
            {
                key.split(".", 1)[0]
                for key in state.keys()
                if isinstance(key, str) and "." in key
            }
        )
        self._proj_by_name = nn.ModuleDict()
        for root in roots:
            w0 = state.get(f"{root}.0.weight")
            w2 = state.get(f"{root}.2.weight")
            if not torch.is_tensor(w0) or not torch.is_tensor(w2):
                continue
            in_dim = int(w0.size(1))
            hidden_dim = int(w0.size(0))
            out_dim = int(w2.size(0))
            proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )
            self._set_requires_grad(proj, False)
            self._proj_by_name[root] = proj
        if self._proj_by_name:
            self._proj_by_name.load_state_dict(dict(state), strict=False)

    def state_dict(self) -> dict[str, Any]:
        out = super().state_dict()
        out["proj_by_name"] = self._state_to_cpu(self._proj_by_name.state_dict())
        out["local_param_ids_by_name"] = {
            str(key): [int(v) for v in value]
            for key, value in self._local_param_ids_by_name.items()
        }
        out["local_opt_state_by_name"] = {
            str(name): self._state_to_cpu(opt.state_dict())
            for name, opt in self._local_opt_by_name.items()
            if isinstance(opt, torch.optim.Optimizer)
        }
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._local_opt_by_name = {}
        self._pending_local_opt_state_by_name = {}

        raw_proj = state.get("proj_by_name", {})
        if isinstance(raw_proj, Mapping):
            self._rebuild_proj_modules_from_state(raw_proj)

        raw_param_ids = state.get("local_param_ids_by_name", {})
        self._local_param_ids_by_name = {}
        if isinstance(raw_param_ids, Mapping):
            for key, value in raw_param_ids.items():
                if isinstance(value, (tuple, list)):
                    self._local_param_ids_by_name[str(key)] = tuple(int(v) for v in value)

        raw_opt = state.get("local_opt_state_by_name", {})
        if isinstance(raw_opt, Mapping):
            for key, opt_state in raw_opt.items():
                if isinstance(opt_state, Mapping):
                    self._pending_local_opt_state_by_name[str(key)] = self._state_to_cpu(dict(opt_state))

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
            pending = self._pending_local_opt_state_by_name.pop(name, None)
            if isinstance(pending, Mapping):
                opt.load_state_dict(dict(pending))
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
        pending = self._pending_local_opt_state_by_name.pop(name, None)
        if isinstance(pending, Mapping):
            opt.load_state_dict(dict(pending))
            self._set_optimizer_lr(opt, lr_here)
        return proj, opt

    @staticmethod
    def _build_extended_attention_mask(model: nn.Module, batch: Mapping[str, Any], device: torch.device):
        attn = batch.get("attention_mask", None)
        if attn is None or (not torch.is_tensor(attn)):
            return None

        input_shape = None
        if "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
            input_shape = batch["input_ids"].shape
        else:
            input_shape = attn.shape

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
            try:
                out = module(xin, attention_mask=attention_mask)
                return self._unpack_block_output(out)
            except TypeError:
                pass
            try:
                out = module(xin, src_key_padding_mask=attention_mask)
                return self._unpack_block_output(out)
            except TypeError:
                pass

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

    @contextmanager
    def _temporarily_freeze_except(self, model: nn.Module, keep_params: Sequence[nn.Parameter]):
        keep_ids = {id(p) for p in keep_params}
        saved: list[tuple[nn.Parameter, bool]] = []
        for p in model.parameters():
            prev = bool(p.requires_grad)
            saved.append((p, prev))
            p.requires_grad = (id(p) in keep_ids)
        try:
            yield
        finally:
            for p, prev in saved:
                p.requires_grad = prev

    # ============================================================
    # faithful local updates
    # ============================================================

    def _local_supcon_update_linear(
        self,
        *,
        layer: nn.Linear,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        if xin.dim() != 2:
            raise RuntimeError(f"SCL linear block expected 2D input, got {tuple(xin.shape)}.")

        was_req = [p.requires_grad for p in layer.parameters()]
        self._set_requires_grad(layer, True)
        self._set_requires_grad(proj, False)

        h = layer(xin.detach())
        z = proj(h)
        loss = self._supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for p, prev in zip(layer.parameters(), was_req):
            p.requires_grad = prev

        return float(loss.detach().item())

    def _local_supcon_update_feature_block(
        self,
        *,
        block: nn.Module,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        if xin.dim() != 4:
            raise RuntimeError(f"SCL feature block expected 4D input, got {tuple(xin.shape)}.")

        was_req = [p.requires_grad for p in block.parameters()]
        self._set_requires_grad(block, True)
        self._set_requires_grad(proj, False)

        h = self._run_block(block, xin.detach(), attention_mask=None)
        if not torch.is_tensor(h) or h.dim() != 4:
            raise RuntimeError(f"SCL feature block must output 4D tensor, got {type(h)} / {getattr(h, 'shape', None)}")

        v = h.mean(dim=(2, 3))  # faithful V1 behavior: GAP
        z = proj(v)
        loss = self._supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for p, prev in zip(block.parameters(), was_req):
            p.requires_grad = prev

        return float(loss.detach().item())

    def _local_supcon_update_transformer_block(
        self,
        *,
        block: nn.Module,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,
        labels: torch.Tensor,
        rep: str,
        attention_mask: torch.Tensor | None,
    ) -> float:
        if xin.dim() != 3:
            raise RuntimeError(f"SCL transformer block expected 3D input, got {tuple(xin.shape)}.")

        was_req = [p.requires_grad for p in block.parameters()]
        self._set_requires_grad(block, True)
        self._set_requires_grad(proj, False)

        h = self._run_block(block, xin.detach(), attention_mask=attention_mask)
        v = self._select_representation(h, rep=rep)
        if v.dim() != 2:
            raise RuntimeError(f"SCL transformer representation must be 2D, got {tuple(v.shape)}.")

        z = proj(v)
        loss = self._supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for p, prev in zip(block.parameters(), was_req):
            p.requires_grad = prev

        return float(loss.detach().item())

    # ============================================================
    # main
    # ============================================================

    def train_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        model.train()

        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y = b.get("labels", None)
            if not torch.is_tensor(y):
                raise RuntimeError("SCL mapping batches must contain tensor key 'labels'.")
            labels = self._label_indices(y)
            ext_attention_mask = self._build_extended_attention_mask(model, b, device=torch.device(device))
        else:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise RuntimeError(f"SCL expects batch=(x, y) or mapping batch, got {type(batch)}.")
            x, y = batch
            x = to_device(x, device)
            y = to_device(y, device)
            if not torch.is_tensor(y):
                raise RuntimeError(f"SCL expects tensor labels, got {type(y)}.")
            labels = self._label_indices(y)
            b = None
            ext_attention_mask = None

        # ------------------------------------------------------------
        # 1) Snapshot forward with cache + head-only global update
        #    (faithful to the first implementation)
        # ------------------------------------------------------------
        raw_blocks = self._declared_block_specs(model)
        if raw_blocks:
            head_params: list[nn.Parameter] = []
            seen: set[int] = set()
            for block in raw_blocks:
                if not bool(getattr(block, "is_output", False)):
                    continue
                module = getattr(block, "module", None)
                if not isinstance(module, nn.Module):
                    continue
                for p in module.parameters():
                    if id(p) not in seen:
                        head_params.append(p)
                        seen.add(id(p))
        else:
            head_params = list(model.parameters())

        with self._temporarily_freeze_except(model, head_params):
            if isinstance(batch, Mapping):
                out, cache, views = forward_with_standard_cache(
                    model,
                    cache_spec=self._cache_spec_for_model(model),
                    **b,
                )
            else:
                out, cache, views = forward_with_standard_cache(
                    model,
                    x,
                    cache_spec=self._cache_spec_for_model(model),
                )

            self.zero_grad()
            loss, stats = self._best_effort_stats(task, out, y)
            loss.backward()
            self.step(head_params, require_grads=True, check_finite_grads=True)

        # ------------------------------------------------------------
        # 2) Recover snapshot block inputs from cache
        # ------------------------------------------------------------
        if not isinstance(views, Mapping):
            raise RuntimeError("SCL requires views returned by forward_with_standard_cache().")

        blocks, output_idx = self._ordered_training_blocks(views)
        hidden_blocks = blocks[:output_idx]
        block_inputs = self._extract_block_inputs(cache=cache, views=views)

        # ------------------------------------------------------------
        # 3) Local SCL updates on hidden blocks, using snapshot inputs
        # ------------------------------------------------------------
        local_losses: list[float] = []
        depth = 0

        for block in hidden_blocks:
            name = str(block.get("name", ""))
            module = block["module"]
            rep = str(block.get("rep", "identity"))

            xin = block_inputs.get(name, None)
            if not torch.is_tensor(xin):
                continue
            if not torch.is_floating_point(xin):
                continue

            # Dense block (Linear or local chain ending in a 2D representation)
            if xin.dim() == 2:
                proj, opt = self._ensure_local_supcon_modules(
                    name=name,
                    module=module,
                    device=torch.device(device),
                    depth=depth,
                )
                loss_local = self._local_supcon_update_dense_block(
                    block=module,
                    proj=proj,
                    opt=opt,
                    xin=xin,
                    labels=labels,
                )
                local_losses.append(loss_local)
                depth += 1
                continue

            # Transformer-like block
            if xin.dim() == 3:
                proj, opt = self._ensure_local_supcon_modules(
                    name=name,
                    module=module,
                    device=torch.device(device),
                    depth=depth,
                    feat_dim_override=int(xin.size(-1)),
                )
                loss_local = self._local_supcon_update_transformer_block(
                    block=module,
                    proj=proj,
                    opt=opt,
                    xin=xin,
                    labels=labels,
                    rep=rep,
                    attention_mask=ext_attention_mask,
                )
                local_losses.append(loss_local)
                depth += 1
                continue

            # Feature block
            if xin.dim() == 4:
                proj, opt = self._ensure_local_supcon_modules(
                    name=name,
                    module=module,
                    device=torch.device(device),
                    depth=depth,
                )
                loss_local = self._local_supcon_update_feature_block(
                    block=module,
                    proj=proj,
                    opt=opt,
                    xin=xin,
                    labels=labels,
                )
                local_losses.append(loss_local)
                depth += 1
                continue

            # else: ignore unsupported input rank

        out_stats = dict(stats)
        out_stats["loss"] = float(loss.detach().item())
        out_stats["supcon_loss"] = float(sum(local_losses) / len(local_losses)) if local_losses else 0.0
        self._mark_step_done()
        return out_stats
