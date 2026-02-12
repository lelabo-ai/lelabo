# lab/algorithms/update_rules/softhebb_demo_conv.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule
from ...core.batch import to_device
from ...core.steps import maybe_accuracy_from_logits

from ...models.deep_softhebb import SoftHebbBlock


class CustomStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, nb_epochs: int):
        self.nb_epochs = int(nb_epochs)
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_threshold = {int(self.nb_epochs * r) for r in threshold_ratios}
        super().__init__(optimizer)

    def get_lr(self):
        # IMPORTANT: utiliser le LR courant (comme l'officiel), pas base_lrs
        if self.last_epoch in self.step_threshold:
            return [group["lr"] * 0.5 for group in self.optimizer.param_groups]
        return [group["lr"] for group in self.optimizer.param_groups]



class SoftHebb(UpdateRule):
    """
    Two-phase version of your demo:

      Phase A (unsupervised): epochs [0 .. unsup_epochs-1]
        - update conv blocks with *demo* SoftHebb update
        - do NOT train head

      Phase B (supervised): epochs [unsup_epochs .. end]
        - freeze conv blocks (and set conv+bn to eval)
        - train head with backprop
        - do NOT do SoftHebb updates
    """

    def __init__(
        self,
        # demo-like per-block base learning rates (positive magnitudes)
        lr_conv1: float = 0.08,
        lr_conv2: float = 0.005,
        lr_conv3: float = 0.01,
        power_lr: float = 0.5,
        # phase split
        unsup_epochs: int = 1,          # demo does ~1 epoch pass
        sup_epochs: int = 50,           # demo trains head for 50
        steps_per_epoch: Optional[int] = None,  # only needed if trainer doesn't call epoch hooks
        # head optimizer
        head_lr: float = 1e-3,
        head_weight_decay: float = 0.0,
        eps_norm: float = 1e-10,
    ):
        super().__init__()

        self.lr_by_name = {
            "conv1": float(lr_conv1),
            "conv2": float(lr_conv2),
            "conv3": float(lr_conv3),
        }
        self.power_lr = float(power_lr)
        self.eps_norm = float(eps_norm)

        self.unsup_epochs = int(unsup_epochs)
        self.sup_epochs = int(sup_epochs)
        self.steps_per_epoch = None if steps_per_epoch is None else int(steps_per_epoch)

        self.head_lr = float(head_lr)
        self.head_weight_decay = float(head_weight_decay)

        self._head_optimizer: Optional[torch.optim.Optimizer] = None
        self._head_scheduler: Optional[CustomStepLR] = None
        self._head_param_ids: Optional[Tuple[int, ...]] = None

        # epoch tracking
        self._epoch: int = 0

        # freeze bookkeeping
        self._param_by_id: Optional[Dict[int, nn.Parameter]] = None
        self._conv_frozen_for_sup: bool = False
        self._head_frozen_for_unsup: bool = False

    # ---------------------------
    # Epoch hooks (use if trainer calls them)
    # ---------------------------

    def on_epoch_start(self, epoch: int):
        self._epoch = int(epoch)

    def on_epoch_end(self, epoch: int):
        self._epoch = int(epoch)
        if self._head_scheduler is not None and self._epoch >= self.unsup_epochs:
            sup_ep = self._epoch - self.unsup_epochs
            self._head_scheduler.step(sup_ep)

    def _infer_epoch_if_needed(self) -> int:
        # If epoch hooks aren't used, try to infer from global_step
        if self.steps_per_epoch is None or self.steps_per_epoch <= 0:
            return self._epoch
        return int(self.global_step // self.steps_per_epoch)

    def _in_unsup_phase(self) -> bool:
        ep = self._infer_epoch_if_needed()
        return ep < self.unsup_epochs

    # ---------------------------
    # Param freezing helpers
    # ---------------------------

    def _ensure_param_index(self, model: nn.Module) -> None:
        if self._param_by_id is None:
            self._param_by_id = {id(p): p for p in model.parameters()}

    def _freeze_head_once_for_unsup(self, model: nn.Module, blocks) -> None:
        """
        During unsup, freeze *everything* (including head) since we update conv manually.
        This matches "no grad" behavior and avoids autograd overhead.
        """
        if self._head_frozen_for_unsup:
            return
        self._ensure_param_index(model)
        for p in self._param_by_id.values():
            p.requires_grad = False
        self._head_frozen_for_unsup = True

    def _freeze_conv_once_for_sup(self, model: nn.Module, blocks) -> None:
        """
        During supervised phase, freeze conv blocks and set them (and their BN) to eval(),
        while keeping head trainable.
        """
        if self._conv_frozen_for_sup:
            return

        # freeze all first, then unfreeze head
        self._ensure_param_index(model)
        for p in self._param_by_id.values():
            p.requires_grad = False

        # set conv blocks to eval (like your demo: conv + bn eval)
        for b in blocks:
            if getattr(b, "is_output", False):
                continue
            mod = getattr(b, "module", None)
            if isinstance(mod, SoftHebbBlock):
                mod.eval()          # sets bn + any submodules to eval
                mod.bn.eval()
                mod.conv.eval()

        # unfreeze head params
        head_params: List[nn.Parameter] = []
        for b in blocks:
            if not getattr(b, "is_output", False):
                continue
            for p in b.module.parameters():
                head_params.append(p)
                p.requires_grad = True

        self._conv_frozen_for_sup = True

    # ---------------------------
    # Head optimizer setup
    # ---------------------------

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

    def _ensure_head_optim(self, head_params: Sequence[nn.Parameter]):
        if not head_params:
            self._head_optimizer = None
            self._head_scheduler = None
            self._head_param_ids = None
            return

        param_ids = tuple(sorted(id(p) for p in head_params))
        if self._head_optimizer is not None and self._head_param_ids == param_ids:
            return

        self._head_param_ids = param_ids
        self._head_optimizer = torch.optim.Adam(
            head_params, lr=self.head_lr, weight_decay=self.head_weight_decay
        )
        self._head_scheduler = CustomStepLR(self._head_optimizer, nb_epochs=self.sup_epochs)

    # ---------------------------
    # Demo SoftHebb conv update
    # ---------------------------

    def _lr_tensor_from_weight(self, W: torch.Tensor, base_lr: float) -> torch.Tensor:
        w2d = W.view(W.shape[0], -1)
        norm_diff = torch.abs(torch.linalg.norm(w2d, dim=1, ord=2) - 1.0) + self.eps_norm
        lr_vec = base_lr * (norm_diff ** self.power_lr)  # [Cout]
        return lr_vec[:, None, None, None]               # [Cout,1,1,1]

    @torch.no_grad()
    def _demo_softhebb_delta(self, block: SoftHebbBlock, x_in: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # matches your demo: bn -> reflect pad -> conv already produced u (weighted_input)
        x_bn = block.bn(x_in)
        x = block.conv.padded_input(x_bn)

        weighted_input = u
        B, OC, OH, OW = weighted_input.shape

        flat_weighted = weighted_input.transpose(0, 1).reshape(OC, -1)
        flat_soft = torch.softmax(block.conv.t_invert * flat_weighted, dim=0)
        flat_soft = -flat_soft
        win = torch.argmax(flat_weighted, dim=0)
        idx = torch.arange(flat_weighted.size(1), device=flat_weighted.device)
        flat_soft[win, idx] = -flat_soft[win, idx]
        softwta = flat_soft.view(OC, B, OH, OW).transpose(0, 1)

        yx = F.conv2d(
            x.transpose(0, 1),
            softwta.transpose(0, 1),
            padding=0,
            stride=block.conv.dilation,
            dilation=block.conv.stride,
            groups=1,
        ).transpose(0, 1)

        yu = torch.sum(softwta * weighted_input, dim=(0, 2, 3))  # (OC,)
        delta = yx - yu.view(-1, 1, 1, 1) * block.conv.weight
        delta.div_(torch.abs(delta).amax() + 1e-30)
        return delta

    @torch.no_grad()
    def _apply_conv_update(self, block: SoftHebbBlock, name: str, x_in: torch.Tensor, u: torch.Tensor):
        base_lr = self.lr_by_name.get(name, None)
        if base_lr is None:
            return
        dW = self._demo_softhebb_delta(block, x_in, u)
        lr_t = self._lr_tensor_from_weight(block.conv.weight, base_lr)
        block.conv.weight.add_(lr_t * dW)

    # ---------------------------
    # Loss helper
    # ---------------------------

    def _loss_from_outputs(self, task, out: Any, y: Any) -> tuple[torch.Tensor, Dict[str, float]]:
        stats: Dict[str, float] = {}
        logits = out.logits if hasattr(out, "logits") else out
        res = task.loss(logits, y)

        if torch.is_tensor(res):
            loss = res
        elif isinstance(res, tuple) and len(res) == 2 and torch.is_tensor(res[0]) and isinstance(res[1], Mapping):
            loss = res[0]
            stats.update({k: float(v) for k, v in res[1].items() if isinstance(v, (int, float))})
        elif isinstance(res, Mapping) and "loss" in res:
            loss = res["loss"]
            if not torch.is_tensor(loss):
                loss = torch.tensor(float(loss), device=logits.device)
            stats.update({k: float(v) for k, v in res.items() if k != "loss" and isinstance(v, (int, float))})
        else:
            raise TypeError(f"Unsupported loss return type from task.loss: {type(res)}")

        if torch.is_tensor(logits) and y is not None and torch.is_tensor(y):
            stats["acc"] = maybe_accuracy_from_logits(logits, y)
        return loss, stats

    # ---------------------------
    # Main
    # ---------------------------

    def train_step(self, model, task, batch, device, ep) -> Dict[str, float]:
        if not hasattr(model, "get_blocks"):
            raise RuntimeError("SoftHebbDemoConv requires model.get_blocks().")

        blocks = model.get_blocks()
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError("Expected non-empty list from model.get_blocks().")

        # Move batch once
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            y = b.get("labels", None)
            x = b.get("x", None)
        else:
            x, y = batch
            x = to_device(x, device)
            y = to_device(y, device)

        in_unsup = self._in_unsup_phase()

        # -------------------------
        # Phase A: Unsupervised
        # -------------------------
        if ep <= self.unsup_epochs:
            # Freeze everything (including head), run inference_mode, update conv only
            self._freeze_head_once_for_unsup(model, blocks)
            model.train()  # keep "train" overall; BN affine=False anyway, but we use inference_mode
            with torch.inference_mode():
                if isinstance(batch, Mapping):
                    out, cache = model(return_cache=True, **b)
                else:
                    out, cache = model(x, return_cache=True)

            # apply conv updates
            if not isinstance(cache, Mapping) or "block_inputs" not in cache or "block_outputs" not in cache:
                raise RuntimeError("Expected cache with keys: 'block_inputs' and 'block_outputs'.")

            with torch.no_grad():
                for bspec in blocks:
                    if getattr(bspec, "is_output", False):
                        continue
                    name = getattr(bspec, "name", None)
                    mod = getattr(bspec, "module", None)
                    if name is None or not isinstance(mod, SoftHebbBlock):
                        continue
                    xin = cache["block_inputs"].get(name, None)
                    u = cache["block_outputs"].get(name, None)
                    if torch.is_tensor(xin) and torch.is_tensor(u):
                        self._apply_conv_update(mod, name, xin, u)

            # optional stats: no supervised loss in unsup phase
            out_stats: Dict[str, float] = {"loss": 0.0}
            self.global_step += 1
            return out_stats

        # -------------------------
        # Phase B: Supervised
        # -------------------------
        # Freeze convs once and keep head trainable; disable SoftHebb updates.
        self._freeze_conv_once_for_sup(model, blocks)

        head_params = self._collect_head_params(blocks)
        self._ensure_head_optim(head_params)

        model.train()
        # Make sure conv blocks stay eval (like demo); head remains trainable
        for b in blocks:
            if getattr(b, "is_output", False):
                continue
            mod = getattr(b, "module", None)
            if isinstance(mod, SoftHebbBlock):
                mod.eval()

        # forward with grads (only head params require_grad=True)
        if isinstance(batch, Mapping):
            out, cache = model(return_cache=True, **b)
        else:
            out, cache = model(x, return_cache=True)

        stats: Dict[str, float] = {}
        loss_t: Optional[torch.Tensor] = None

        if y is not None and self._head_optimizer is not None:
            loss, stats = self._loss_from_outputs(task, out, y)
            self._head_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._head_optimizer.step()
            loss_t = loss.detach()
        else:
            # if no labels, just return 0 loss
            loss_t = None

        out_stats = dict(stats)
        out_stats["loss"] = float(loss_t.item()) if loss_t is not None else 0.0
        if ep > self._epoch:
            self.on_epoch_end(ep) 
            self._epoch = ep
        self.global_step += 1
        return out_stats
