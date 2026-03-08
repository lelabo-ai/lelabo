from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OptimizerUpdateRule
from ...core.batch import to_device
from ...core.steps import compute_loss_and_stats
from ...models.cache_provider import CacheSpec, forward_with_standard_cache

try:
    from ...models.builtins.deep_softhebb import SoftHebbBlock as _SoftHebbBlock
except Exception:  # pragma: no cover - optional model dependency
    _SoftHebbBlock = None


class SoftHebb(OptimizerUpdateRule):
    """
    Minimal SoftHebb update rule:
    - Unsupervised phase (epoch <= unsup_epochs): local Hebbian updates on hidden blocks.
    - Supervised phase (epoch > unsup_epochs): freeze hidden blocks, train head with BP.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        head_optimizer: torch.optim.Optimizer | None = None,
        base_lr: float = 0.01,
        lr_conv1: float = 0.08,
        lr_conv2: float = 0.005,
        lr_conv3: float = 0.01,
        power_lr: float = 0.5,
        unsup_epochs: int = 1,
        sup_epochs: int = 50,
        steps_per_epoch: int | None = None,
        eps_norm: float = 1e-10,
        conv_t_invert: float = 12.0,
        grad_clip: float | None = None,
    ) -> None:
        chosen_optim = head_optimizer if head_optimizer is not None else optimizer
        super().__init__(
            optimizer=chosen_optim,
            grad_clip=grad_clip,
            strict_require_grads=True,
            check_finite_grads=True,
        )
        self._head_optimizer = chosen_optim

        self.base_lr = float(base_lr)
        self.lr_by_name = {
            "conv1": float(lr_conv1),
            "conv2": float(lr_conv2),
            "conv3": float(lr_conv3),
        }
        self.power_lr = float(power_lr)
        self.unsup_epochs = int(unsup_epochs)
        self.sup_epochs = int(sup_epochs)
        self.steps_per_epoch = None if steps_per_epoch is None else int(steps_per_epoch)
        self.eps_norm = float(eps_norm)
        self.conv_t_invert = float(conv_t_invert)

    @staticmethod
    def _is_softhebb_block(module: nn.Module) -> bool:
        return (_SoftHebbBlock is not None) and isinstance(module, _SoftHebbBlock)

    def _cache_spec(self) -> CacheSpec:
        param_types: list[type[nn.Module]] = [nn.Linear, nn.Conv2d]
        if _SoftHebbBlock is not None:
            param_types.insert(0, _SoftHebbBlock)
        return CacheSpec(
            trainable_module_types=tuple(param_types),
            capture_inputs=True,
            capture_outputs=True,
            require_single_call=True,
            require_single_output_head=True,
        )

    def _current_epoch(self, state: Any) -> int:
        if state is not None and hasattr(state, "epoch"):
            try:
                return int(getattr(state, "epoch"))
            except Exception:
                pass
        if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
            return int(self.global_step // self.steps_per_epoch) + 1
        return 1

    def _in_unsup_phase(self, state: Any) -> bool:
        return self._current_epoch(state) <= self.unsup_epochs

    @staticmethod
    def _dedup_params(params: Sequence[nn.Parameter]) -> list[nn.Parameter]:
        out: list[nn.Parameter] = []
        seen: set[int] = set()
        for p in params:
            if not isinstance(p, nn.Parameter):
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
        return out

    def _ensure_head_optim(self, head_params: Sequence[nn.Parameter]) -> None:
        if not head_params:
            self._head_optimizer = None
            return
        if self._head_optimizer is None:
            raise RuntimeError(
                "SoftHebb requires an optimizer provided by the runner. "
                "Pass head_optimizer or optimizer at construction."
            )

    @torch.no_grad()
    def _lr_tensor_from_weight(self, weight: torch.Tensor, base_lr: float) -> torch.Tensor:
        w2d = weight.view(weight.shape[0], -1)
        norm_diff = torch.abs(torch.linalg.norm(w2d, dim=1, ord=2) - 1.0) + self.eps_norm
        lr_vec = float(base_lr) * (norm_diff ** self.power_lr)
        return lr_vec[:, None, None, None]

    @torch.no_grad()
    def _lr_tensor_from_linear_weight(self, weight: torch.Tensor, base_lr: float) -> torch.Tensor:
        w2d = weight.view(weight.shape[0], -1)
        norm_diff = torch.abs(torch.linalg.norm(w2d, dim=1, ord=2) - 1.0) + self.eps_norm
        lr_vec = float(base_lr) * (norm_diff ** self.power_lr)
        return lr_vec[:, None]

    @staticmethod
    @torch.no_grad()
    def _normalize_update(delta: torch.Tensor) -> torch.Tensor:
        delta = delta.clone()
        delta.div_(torch.abs(delta).amax() + 1e-30)
        return delta

    @torch.no_grad()
    def _demo_softhebb_delta_linear(
        self,
        layer: nn.Linear,
        x_in: torch.Tensor,
        preact: torch.Tensor,
        *,
        t_invert: float,
    ) -> torch.Tensor:
        if x_in.dim() != 2 or preact.dim() != 2:
            raise RuntimeError(f"SoftHebb Linear expects 2D x/u, got {tuple(x_in.shape)} and {tuple(preact.shape)}.")
        if int(x_in.size(0)) != int(preact.size(0)):
            raise RuntimeError("SoftHebb Linear batch mismatch between x and pre-activation.")
        if int(preact.size(1)) != int(layer.weight.size(0)):
            raise RuntimeError("SoftHebb Linear pre-activation width mismatch with layer out_features.")

        flat_u = preact.t().contiguous()  # [out, B]
        flat_soft = torch.softmax(float(t_invert) * flat_u, dim=0)
        flat_soft = -flat_soft
        winner = torch.argmax(flat_u, dim=0)
        idx = torch.arange(flat_u.size(1), device=flat_u.device)
        flat_soft[winner, idx] = -flat_soft[winner, idx]
        wta = flat_soft.t().contiguous()  # [B, out]

        yx = wta.t() @ x_in
        yu = torch.sum(wta * preact, dim=0)
        delta = yx - yu[:, None] * layer.weight
        return self._normalize_update(delta)

    @torch.no_grad()
    def _demo_softhebb_delta_conv2d(
        self,
        conv: nn.Conv2d,
        x_in: torch.Tensor,
        preact: torch.Tensor,
        *,
        t_invert: float,
    ) -> torch.Tensor:
        if int(conv.groups) != 1:
            raise RuntimeError("SoftHebb Conv2d update currently supports groups=1 only.")
        if x_in.dim() != 4 or preact.dim() != 4:
            raise RuntimeError("SoftHebb Conv2d expects 4D x/u tensors.")

        _, out_channels, out_h, out_w = preact.shape
        flat_u = preact.transpose(0, 1).reshape(out_channels, -1)
        flat_soft = torch.softmax(float(t_invert) * flat_u, dim=0)
        flat_soft = -flat_soft
        winner = torch.argmax(flat_u, dim=0)
        idx = torch.arange(flat_u.size(1), device=flat_u.device)
        flat_soft[winner, idx] = -flat_soft[winner, idx]
        soft_wta = flat_soft.view(out_channels, preact.size(0), out_h, out_w).transpose(0, 1)

        yx = F.conv2d(
            x_in.transpose(0, 1),
            soft_wta.transpose(0, 1),
            padding=conv.padding,
            stride=conv.dilation,
            dilation=conv.stride,
            groups=1,
        ).transpose(0, 1)
        k_h, k_w = conv.kernel_size
        yx = yx[:, :, :k_h, :k_w]

        yu = torch.sum(soft_wta * preact, dim=(0, 2, 3))
        delta = yx - yu.view(-1, 1, 1, 1) * conv.weight
        return self._normalize_update(delta)

    @torch.no_grad()
    def _demo_softhebb_delta_block(self, block: nn.Module, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._is_softhebb_block(block):
            raise TypeError(f"Expected SoftHebbBlock module, got {type(block)}.")
        if x_in.dim() != 4:
            raise RuntimeError("SoftHebbBlock update expects a 4D block input tensor.")

        x_bn = block.bn(x_in)
        padded = block.conv.padded_input(x_bn)
        preact = block.conv(x_bn)

        bsz, out_channels, out_h, out_w = preact.shape
        flat_u = preact.transpose(0, 1).reshape(out_channels, -1)
        flat_soft = torch.softmax(float(block.conv.t_invert) * flat_u, dim=0)
        flat_soft = -flat_soft
        winner = torch.argmax(flat_u, dim=0)
        idx = torch.arange(flat_u.size(1), device=flat_u.device)
        flat_soft[winner, idx] = -flat_soft[winner, idx]
        soft_wta = flat_soft.view(out_channels, bsz, out_h, out_w).transpose(0, 1)

        yx = F.conv2d(
            padded.transpose(0, 1),
            soft_wta.transpose(0, 1),
            padding=0,
            stride=block.conv.dilation,
            dilation=block.conv.stride,
            groups=1,
        ).transpose(0, 1)
        yu = torch.sum(soft_wta * preact, dim=(0, 2, 3))
        delta = yx - yu.view(-1, 1, 1, 1) * block.conv.weight
        return self._normalize_update(delta), preact

    @torch.no_grad()
    def _apply_unsup_update_for_block(self, *, name: str, module: nn.Module, x_in: torch.Tensor, u: torch.Tensor) -> None:
        base_name = str(name).split(".")[-1]
        base_lr = float(self.lr_by_name.get(base_name, self.base_lr))

        if self._is_softhebb_block(module):
            delta, _ = self._demo_softhebb_delta_block(module, x_in)
            lr_t = self._lr_tensor_from_weight(module.conv.weight, base_lr)
            module.conv.weight.add_(lr_t * delta)
            return

        if isinstance(module, nn.Conv2d):
            delta = self._demo_softhebb_delta_conv2d(module, x_in, u, t_invert=self.conv_t_invert)
            lr_t = self._lr_tensor_from_weight(module.weight, base_lr)
            module.weight.add_(lr_t * delta)
            return

        if isinstance(module, nn.Linear):
            delta = self._demo_softhebb_delta_linear(module, x_in, u, t_invert=self.conv_t_invert)
            lr_t = self._lr_tensor_from_linear_weight(module.weight, base_lr)
            module.weight.add_(lr_t * delta)
            return

        raise TypeError(f"Unsupported hidden block type for SoftHebb update: {type(module)}")

    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = bool(flag)

    def _configure_unsup_phase(self, model: nn.Module) -> None:
        for p in model.parameters():
            p.requires_grad = False

    def _configure_sup_phase(
        self,
        model: nn.Module,
        hidden_blocks: Sequence[Mapping[str, Any]],
        output_block: Mapping[str, Any],
    ) -> list[nn.Parameter]:
        for p in model.parameters():
            p.requires_grad = False

        output_module = output_block.get("module")
        if not isinstance(output_module, nn.Module):
            raise RuntimeError("SoftHebb output_block must expose a valid nn.Module under key 'module'.")

        head_params = self._dedup_params(list(output_module.parameters()))
        for p in head_params:
            p.requires_grad = True

        for block in hidden_blocks:
            module = block.get("module")
            if isinstance(module, nn.Module):
                module.eval()
                if self._is_softhebb_block(module):
                    module.bn.eval()
                    module.conv.eval()
        return head_params

    def _unsup_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        model.train()
        self._configure_unsup_phase(model)

        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            x_args = ()
            x_kwargs = dict(b)
        else:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise RuntimeError(f"SoftHebb expects batch=(x, y) or mapping batch, got {type(batch)}.")
            x, _y = batch
            x_args = (to_device(x, device),)
            x_kwargs = {}

        with torch.inference_mode():
            _out, _cache, views = forward_with_standard_cache(
                model,
                *x_args,
                cache_spec=self._cache_spec(),
                **x_kwargs,
            )

        execution_blocks = views.get("execution_blocks", [])
        hidden_blocks = [
            b for b in execution_blocks
            if bool(b.get("is_trainable", False)) and not bool(b.get("is_output", False))
        ]

        with torch.no_grad():
            for block in hidden_blocks:
                name = str(block.get("name", ""))
                module = block.get("module")
                if not isinstance(module, nn.Module):
                    continue
                x_in = block.get("x")
                u = block.get("u")
                if not torch.is_tensor(x_in):
                    raise RuntimeError(f"SoftHebb missing tensor input cache for block '{name}'.")
                if self._is_softhebb_block(module):
                    # SoftHebbBlock pre-activation is recomputed from block internals.
                    u = torch.empty(0, device=x_in.device)  # placeholder, ignored by block path
                if not torch.is_tensor(u):
                    raise RuntimeError(f"SoftHebb missing tensor output cache for block '{name}'.")
                self._apply_unsup_update_for_block(name=name, module=module, x_in=x_in, u=u)

        return {"loss": 0.0}

    def _sup_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        model.train()

        # Inspect blocks using cache provider so head detection follows v2 conventions.
        if isinstance(batch, Mapping):
            b = to_device(batch, device)
            x_args = ()
            x_kwargs = dict(b)
        else:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise RuntimeError(f"SoftHebb expects batch=(x, y) or mapping batch, got {type(batch)}.")
            x, _y = batch
            x_args = (to_device(x, device),)
            x_kwargs = {}

        with torch.inference_mode():
            _out, _cache, views = forward_with_standard_cache(
                model,
                *x_args,
                cache_spec=self._cache_spec(),
                **x_kwargs,
            )

        execution_blocks = views.get("execution_blocks", [])
        output_blocks = views.get("output_blocks", [])
        if not isinstance(output_blocks, list):
            output_blocks = []
        output_blocks = [b for b in output_blocks if isinstance(b, Mapping)]
        if len(output_blocks) != 1:
            raise RuntimeError(
                "SoftHebb supervised phase requires exactly one output head "
                f"in views['output_blocks'], got {len(output_blocks)}."
            )
        output_block = output_blocks[0]
        hidden_blocks = [
            b for b in execution_blocks
            if bool(b.get("is_trainable", False)) and not bool(b.get("is_output", False))
        ]

        head_params = self._configure_sup_phase(model, hidden_blocks, output_block)
        self._ensure_head_optim(head_params)
        if not head_params or self._head_optimizer is None:
            return {"loss": 0.0}

        self.zero_grad()
        loss, stats = compute_loss_and_stats(model, task, batch, device)
        loss.backward()
        self.step(head_params, require_grads=True, check_finite_grads=True)

        out = dict(stats)
        out["loss"] = float(loss.item())
        return out

    def train_step(self, model, task, batch, device, state=None) -> dict[str, Any]:
        if self._in_unsup_phase(state):
            out = self._unsup_step(model, task, batch, device, state=state)
        else:
            out = self._sup_step(model, task, batch, device, state=state)
        self._mark_step_done()
        return out
