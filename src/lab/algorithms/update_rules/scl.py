# lab/algorithms/update_rules/scl.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule
from ...core.batch import to_device
from ...core.steps import maybe_accuracy_from_logits
from ...models.deep_softhebb import SoftHebbBlock


class SoftContrastiveLearning(UpdateRule):
    """
    Hybrid rule (local SupCon everywhere in the backbone):

      - Output blocks (bspec.is_output=True): updated with standard backprop (heads-only optimizer).

      - Intermediate nn.Linear blocks: updated with local SupCon with a per-block frozen projection.

      - Intermediate "feature blocks" (nn.Conv2d / torchvision ResNet blocks):
          * compute h = block(xin) -> expects [B, C, H, W]
          * v = GAP(h) -> [B, C]
          * z = proj(v) -> [B, D]
          * SupCon loss -> local backward -> updates ONLY block params (proj frozen)

    Requirements:
      - model.get_blocks() -> list of BlockSpec-like objects, each has:
          - name: str
          - module: nn.Module
          - is_output: bool
      - model.forward(..., return_cache=True) -> (out, cache)
      - cache["block_inputs"][block_name] exists
    """

    def __init__(
        self,
        # heads (backprop)
        head_lr: float = 3e-4,
        head_optim: str = "ano",  # "adamw" | "sgd" | "ano"
        head_weight_decay: float = 0.0,
        head_optimizer: Optional[torch.optim.Optimizer] = None,
        # local SupCon (per intermediate block)
        supcon_tau: float = 0.1,
        local_lr: float = 3e-4,
        local_optim: str = "adamw",  # "adamw" | "sgd"
        local_weight_decay: float = 0.0,
        # projection
        proj_dim: int = 256,
        proj_hidden_dim: Optional[int] = None,
        # depth LR scheduler
        depth_lr_gamma: float = 0.5,
        depth_lr_min_factor: float = 0.01,
        depth_lr_max_factor: float = 100.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = float(eps)

        # --- heads (global supervised)
        self.head_lr = float(head_lr)
        self.head_optim = str(head_optim).lower()
        self.head_weight_decay = float(head_weight_decay)
        self._head_optimizer: Optional[torch.optim.Optimizer] = head_optimizer
        self._head_optimizer_external: bool = head_optimizer is not None
        self._head_param_ids: Optional[Tuple[int, ...]] = None

        # --- local SupCon
        self.supcon_tau = float(supcon_tau)
        self.local_lr = float(local_lr)
        self.local_optim = str(local_optim).lower()
        self.local_weight_decay = float(local_weight_decay)
        self.proj_dim = int(proj_dim)
        self.proj_hidden_dim = proj_hidden_dim

        # --- depth LR scheduler params
        self.depth_lr_gamma = float(depth_lr_gamma)
        self.depth_lr_min_factor = float(depth_lr_min_factor)
        self.depth_lr_max_factor = float(depth_lr_max_factor)

        # projection heads per block
        self._proj_by_name = nn.ModuleDict()

        # per-block local optimizers
        self._local_opt_by_name: Dict[str, torch.optim.Optimizer] = {}
        self._local_param_ids_by_name: Dict[str, Tuple[int, ...]] = {}
        self._local_depth_by_name: Dict[str, int] = {}
        self._local_lr_by_name: Dict[str, float] = {}

        # one-time freeze / param index (for heads)
        self._param_by_id: Optional[Dict[int, nn.Parameter]] = None
        self._frozen_once: bool = True
        self._trainable_ids: set[int] = set()

    # ============================================================
    # SupCon
    # ============================================================

    def supervised_contrastive_loss(
        self,
        z: torch.Tensor,      # [B, D]
        labels: torch.Tensor, # [B]
        tau: float = 0.1,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        B = z.size(0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / tau
        sim = sim - sim.max(dim=1, keepdim=True)[0]

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

    def _unpack_block_output(self, out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            return out.last_hidden_state
        raise RuntimeError(f"Unsupported block output type: {type(out)}")

    def _select_representation(self, h: torch.Tensor, rep: str) -> torch.Tensor:
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
            # spatial -> GAP
            return h.mean(dim=(2, 3))
        return h

    def _build_extended_attention_mask(self, model: nn.Module, batch: Mapping[str, Any], device: torch.device):
        attn = batch.get("attention_mask", None)
        if attn is None or (not torch.is_tensor(attn)):
            return None
        input_shape = None
        if "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
            input_shape = batch["input_ids"].shape
        else:
            input_shape = attn.shape

        # try model.get_extended_attention_mask (HF standard)
        if hasattr(model, "get_extended_attention_mask"):
            try:
                return model.get_extended_attention_mask(attn, input_shape, device=device)
            except TypeError:
                try:
                    return model.get_extended_attention_mask(attn, input_shape)
                except Exception:
                    pass

        # try base model (e.g. HFSequenceClassifier exposes .bert)
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

    # ============================================================
    # Depth LR scheduler helpers
    # ============================================================

    def _depth_scaled_lr(self, depth: int) -> float:
        factor = self.depth_lr_gamma ** int(depth)
        factor = max(self.depth_lr_min_factor, min(self.depth_lr_max_factor, factor))
        return float(self.local_lr * factor)

    def _set_optimizer_lr(self, opt: torch.optim.Optimizer, lr: float) -> None:
        for g in opt.param_groups:
            g["lr"] = lr

    # ============================================================
    # Projection head
    # ============================================================

    def _key(self, name: str) -> str:
        return name.replace(".", "__")

    def _make_projection(self, in_dim: int, device: torch.device) -> nn.Module:
        out_dim = min(self.proj_dim, in_dim) if self.proj_dim > 0 else in_dim
        hidden = self.proj_hidden_dim if self.proj_hidden_dim is not None else in_dim

        proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        ).to(device)

        # frozen params (Variant 1)
        self._set_requires_grad(proj, False)
        return proj

    def _feature_dim_for_module(self, module: nn.Module) -> Optional[int]:
        """
        Donne la dimension "feature" sur laquelle on va faire GAP puis proj.
        """
        if isinstance(module, nn.Linear):
            return int(module.out_features)

        if isinstance(module, SoftHebbBlock):
            return int(module.conv.out_channels)

        if isinstance(module, nn.Conv2d):
            return int(module.out_channels)

        # IMPORTANT: Bottleneck a aussi conv2, donc tester conv3 d'abord
        if hasattr(module, "conv3") and isinstance(getattr(module, "conv3"), nn.Conv2d):
            return int(module.conv3.out_channels)  # Bottleneck
        if hasattr(module, "conv2") and isinstance(getattr(module, "conv2"), nn.Conv2d):
            return int(module.conv2.out_channels)  # BasicBlock

        return None

    def _ensure_local_supcon_modules(
        self,
        name: str,
        module: nn.Module,
        device: torch.device,
        depth: int,
        feat_dim_override: Optional[int] = None,
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        feat_dim = int(feat_dim_override) if feat_dim_override is not None else self._feature_dim_for_module(module)
        if feat_dim is None:
            raise TypeError(f"Unsupported module for local SupCon: {type(module)}")

        k = self._key(name)

        if k not in self._proj_by_name:
            self._proj_by_name[k] = self._make_projection(feat_dim, device)
        else:
            self._proj_by_name[k] = self._proj_by_name[k].to(device)
            self._set_requires_grad(self._proj_by_name[k], False)

        proj = self._proj_by_name[k]

        params = list(module.parameters())
        param_ids = tuple(sorted(id(p) for p in params))

        lr_here = self._depth_scaled_lr(depth)
        self._local_depth_by_name[name] = int(depth)
        self._local_lr_by_name[name] = float(lr_here)

        if name in self._local_opt_by_name and self._local_param_ids_by_name.get(name, None) == param_ids:
            opt = self._local_opt_by_name[name]
            self._set_optimizer_lr(opt, lr_here)
            return proj, opt

        self._local_param_ids_by_name[name] = param_ids

        if self.local_optim == "sgd":
            opt = torch.optim.SGD(params, lr=lr_here, weight_decay=self.local_weight_decay, momentum=0.9)
        else:
            opt = torch.optim.AdamW(params, lr=lr_here, weight_decay=self.local_weight_decay)

        self._local_opt_by_name[name] = opt
        return proj, opt

    # ============================================================
    # Local updates
    # ============================================================

    def _local_supcon_update_linear(
        self,
        layer: nn.Linear,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,     # [B, Din]
        labels: torch.Tensor,  # [B]
    ) -> float:
        was_req = [p.requires_grad for p in layer.parameters()]
        self._set_requires_grad(layer, True)
        self._set_requires_grad(proj, False)

        h = layer(xin)
        z = proj(h)
        loss = self.supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for p, prev in zip(layer.parameters(), was_req):
            p.requires_grad = prev

        return float(loss.detach().item())

    def _local_supcon_update_feature_block(
        self,
        block: nn.Module,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,     # [B, Cin, H, W]
        labels: torch.Tensor,  # [B]
    ) -> float:
        """
        Update générique pour n'importe quel bloc qui renvoie un tenseur 4D.
        (nn.Conv2d, BasicBlock, Bottleneck, etc.)
        """
        was_req = [p.requires_grad for p in block.parameters()]
        self._set_requires_grad(block, True)
        self._set_requires_grad(proj, False)

        h = block(xin)
        if not torch.is_tensor(h) or h.dim() != 4:
            raise RuntimeError(f"Feature block must output 4D tensor, got {type(h)} {getattr(h,'shape',None)}")

        v = h.mean(dim=(2, 3))
        z = proj(v)
        loss = self.supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for p, prev in zip(block.parameters(), was_req):
            p.requires_grad = prev

        return float(loss.detach().item())

    def _call_transformer_layer(self, block: nn.Module, h_in: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        if attention_mask is not None:
            try:
                out = block(h_in, attention_mask=attention_mask)
                return self._unpack_block_output(out)
            except TypeError:
                pass
        out = block(h_in)
        return self._unpack_block_output(out)

    def _local_supcon_update_transformer_block(
        self,
        block: nn.Module,
        proj: nn.Module,
        opt: torch.optim.Optimizer,
        xin: torch.Tensor,     # [B, S, H]
        labels: torch.Tensor,  # [B]
        *,
        rep: str,
        attention_mask: Optional[torch.Tensor],
    ) -> float:
        was_req = [p.requires_grad for p in block.parameters()]
        self._set_requires_grad(block, True)
        self._set_requires_grad(proj, False)

        h_in = xin.detach()
        h = self._call_transformer_layer(block, h_in, attention_mask=attention_mask)
        v = self._select_representation(h, rep=rep)
        z = proj(v)
        loss = self.supervised_contrastive_loss(z, labels, tau=self.supcon_tau, eps=self.eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        for p, prev in zip(block.parameters(), was_req):
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
        if self._head_optimizer_external:
            return
        if not head_params:
            self._head_optimizer = None
            self._head_param_ids = None
            return

        param_ids = tuple(sorted(id(p) for p in head_params))
        if self._head_optimizer is not None and self._head_param_ids == param_ids:
            return

        self._head_param_ids = param_ids

        if self.head_optim == "sgd":
            self._head_optimizer = torch.optim.SGD(head_params, lr=self.head_lr, weight_decay=self.head_weight_decay)
        elif self.head_optim == "ano":
            from ano_optimizer import Ano
            self._head_optimizer = Ano(head_params, lr=self.head_lr, weight_decay=self.head_weight_decay)
        else:
            self._head_optimizer = torch.optim.AdamW(head_params, lr=self.head_lr, weight_decay=self.head_weight_decay)

    # ============================================================
    # One-time freeze for heads
    # ============================================================

    def _ensure_param_index(self, model: nn.Module) -> None:
        if self._param_by_id is None:
            self._param_by_id = {id(p): p for p in model.parameters()}
            self._frozen_once = False
            self._trainable_ids = set()

    def _freeze_non_head_params_once(self, model: nn.Module, head_params: Sequence[nn.Parameter]) -> None:
        self._ensure_param_index(model)
        new_ids = {id(p) for p in head_params}

        if not self._frozen_once:
            for p in self._param_by_id.values():
                p.requires_grad = False
            for pid in new_ids:
                p = self._param_by_id.get(pid, None)
                if p is not None:
                    p.requires_grad = True
            self._trainable_ids = set(new_ids)
            self._frozen_once = True
            return

        if new_ids != self._trainable_ids:
            for pid in (self._trainable_ids - new_ids):
                p = self._param_by_id.get(pid, None)
                if p is not None:
                    p.requires_grad = False
            for pid in (new_ids - self._trainable_ids):
                p = self._param_by_id.get(pid, None)
                if p is not None:
                    p.requires_grad = True
            self._trainable_ids = set(new_ids)

    # ============================================================
    # Loss / stats helper
    # ============================================================

    def _loss_from_outputs(self, task, out: Any, y: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats: Dict[str, float] = {}

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
            raise RuntimeError("SoftContrastiveLearning requires models to expose get_blocks().")

        blocks = model.get_blocks()
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise RuntimeError("SoftContrastiveLearning expects model.get_blocks() to return a non-empty list.")

        # --- Move batch once
        extended_attention_mask = None
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y = b.get("labels", None)
            x_for_forward = None
            extended_attention_mask = self._build_extended_attention_mask(model, b, device=torch.device(device))
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

        # --- Forward (heads) + cache
        can_do_head_bp = (self._head_optimizer is not None) and (y is not None)

        if can_do_head_bp:
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
            raise RuntimeError("SoftContrastiveLearning expects cache['block_inputs'] from model(..., return_cache=True).")

        block_inputs: Mapping[str, Any] = cache["block_inputs"]

        # ============================================================
        # Local SupCon updates for intermediate blocks
        # ============================================================
        local_supcon_losses: List[float] = []
        depth_idx = 0

        for bspec in blocks:
            if getattr(bspec, "is_output", False):
                continue

            if y is None or (not torch.is_tensor(y)):
                continue

            name = getattr(bspec, "name", None)
            mod = getattr(bspec, "module", None)
            if name is None or mod is None:
                continue

            xin = block_inputs.get(name, None)
            if not torch.is_tensor(xin):
                continue

            # Linear blocks: xin expected [B, Din]
            if isinstance(mod, nn.Linear):
                if xin.dim() != 2:
                    continue
                proj, opt = self._ensure_local_supcon_modules(name, mod, device=device, depth=depth_idx)
                loss_local = self._local_supcon_update_linear(mod, proj, opt, xin, y)
                local_supcon_losses.append(loss_local)
                depth_idx += 1
                continue

            # Transformer blocks: xin expected [B, S, H]
            if xin.dim() == 3:
                proj, opt = self._ensure_local_supcon_modules(
                    name,
                    mod,
                    device=device,
                    depth=depth_idx,
                    feat_dim_override=int(xin.size(-1)),
                )
                loss_local = self._local_supcon_update_transformer_block(
                    mod,
                    proj,
                    opt,
                    xin,
                    y,
                    rep=getattr(bspec, "rep", "cls"),
                    attention_mask=extended_attention_mask,
                )
                local_supcon_losses.append(loss_local)
                depth_idx += 1
                continue

            # Feature blocks: xin expected [B, Cin, H, W]
            if xin.dim() == 4:
                proj, opt = self._ensure_local_supcon_modules(name, mod, device=device, depth=depth_idx)
                loss_local = self._local_supcon_update_feature_block(mod, proj, opt, xin, y)
                local_supcon_losses.append(loss_local)
                depth_idx += 1
                continue

            # otherwise ignore
            continue

        out_stats = dict(stats)
        out_stats["loss"] = float(loss_t.item()) if loss_t is not None else 0.0
        out_stats["supcon_loss"] = float(sum(local_supcon_losses) / len(local_supcon_losses)) if local_supcon_losses else 0.0

        self.global_step += 1
        return out_stats
