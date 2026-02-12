# lab/algorithms/update_rules/feedbackalignment.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule
from ...core.batch import to_device


class FeedbackAlignment(UpdateRule):
    """Feedback Alignment (FA) sans autograd, compatible MLP + CNN (Conv2d).

    Approx pratique pour CNN:
      - Conv->Conv avec downsample/pool: on aligne les tailles par upsample nearest.
      - Linear(head)->Conv (GAP): on diffuse le delta sur HxW (avec option / (H*W)).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        feedback_scale: float = 1.0,
        grad_clip: Optional[float] = None,
        grad_scale: float = 1.0,
        pool_backscale: bool = True,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.feedback_scale = float(feedback_scale)
        self.grad_clip = grad_clip
        self.grad_scale = float(grad_scale)
        self.pool_backscale = bool(pool_backscale)

        # feedback par block "suivant" (par nom): B a la même shape que W_next
        self._B: Dict[str, torch.Tensor] = {}
        self._Bsig: Dict[str, Tuple[int, ...]] = {}

    # -------------------------
    # utils
    # -------------------------

    def _relu_deriv(self, z: torch.Tensor) -> torch.Tensor:
        return (z > 0).to(z.dtype)

    def _canon2d(self, t: torch.Tensor) -> torch.Tensor:
        return t if t.dim() == 2 else t.view(t.size(0), -1)

    def _clip_grads(self, params: Sequence[nn.Parameter]):
        if self.grad_clip is None:
            return
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(self.grad_clip))

    def _task_output_deltas(self, task, out: Any, y: Any) -> Dict[str, torch.Tensor]:
        if not hasattr(task, "output_deltas"):
            raise NotImplementedError("FA requires task.output_deltas(out, y).")
        d = task.output_deltas(out, y)
        if not isinstance(d, Mapping):
            raise RuntimeError("task.output_deltas must return dict[str, Tensor].")
        return {k: v for k, v in d.items() if torch.is_tensor(v)}

    def _chain_id(self, block_name: str) -> str:
        n = block_name
        if n.startswith("actor."):
            return "actor"
        if n.startswith("critic."):
            return "critic"
        return "default"

    def _ensure_B(self, name_next: str, mod_next: nn.Module, device, dtype):
        """B_next ~ N(0,0.05) * feedback_scale, shape = W_next.shape."""
        if not hasattr(mod_next, "weight"):
            raise TypeError(f"FA: module '{name_next}' has no weight (type={type(mod_next)}).")
        W = getattr(mod_next, "weight")
        if not torch.is_tensor(W):
            raise TypeError(f"FA: module '{name_next}' weight is not a Tensor.")
        sig = tuple(int(x) for x in W.shape)
        if name_next in self._B and self._Bsig.get(name_next) == sig:
            self._B[name_next] = self._B[name_next].to(device=device, dtype=dtype)
            return
        B = torch.empty(*sig, device=device, dtype=dtype)
        torch.nn.init.normal_(B, mean=0.0, std=0.05)
        B.mul_(self.feedback_scale)
        self._B[name_next] = B
        self._Bsig[name_next] = sig

    def _set_linear_grads(self, layer: nn.Linear, x: torch.Tensor, delta: torch.Tensor):
        dW = delta.T @ x
        db = delta.sum(dim=0)
        if layer.weight.grad is None:
            layer.weight.grad = torch.zeros_like(layer.weight)
        layer.weight.grad.copy_(dW * self.grad_scale)
        if layer.bias is not None:
            if layer.bias.grad is None:
                layer.bias.grad = torch.zeros_like(layer.bias)
            layer.bias.grad.copy_(db * self.grad_scale)

    def _set_conv2d_grads(self, layer: nn.Conv2d, x: torch.Tensor, delta: torch.Tensor):
        if x.dim() != 4 or delta.dim() != 4:
            raise RuntimeError("FA: Conv2d expects 4D x and 4D delta.")
        dW = torch.nn.grad.conv2d_weight(
            x,
            layer.weight.shape,
            delta,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        if layer.weight.grad is None:
            layer.weight.grad = torch.zeros_like(layer.weight)
        layer.weight.grad.copy_(dW * self.grad_scale)

        if layer.bias is not None:
            db = delta.sum(dim=(0, 2, 3))
            if layer.bias.grad is None:
                layer.bias.grad = torch.zeros_like(layer.bias)
            layer.bias.grad.copy_(db * self.grad_scale)

    def _spatial_align(self, delta: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Align spatial dims of delta to target (nearest up/down-sampling)."""
        if delta.dim() != 4:
            return delta
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        if int(delta.size(-2)) == Ht and int(delta.size(-1)) == Wt:
            return delta
        return F.interpolate(delta, size=(Ht, Wt), mode="nearest")

    def _match_head_delta(
        self,
        block_name: str,
        layer: nn.Module,
        deltas: List[Tuple[str, torch.Tensor]],
        used: set,
    ) -> torch.Tensor:
        """Match task deltas to the output head (name hints then shape)."""
        lname = block_name.lower()
        out_dim = None
        if isinstance(layer, nn.Linear):
            out_dim = int(layer.out_features)
        elif isinstance(layer, nn.Conv2d):
            out_dim = int(layer.out_channels)

        # name-hints (RL mainly)
        for k, d in deltas:
            if k in used:
                continue
            if (
                ("logits" in k and ("actor" in lname or "logits" in lname))
                or ("value" in k and ("critic" in lname or "value" in lname))
                or ("q_values" in k and ("q" in lname or "head" in lname))
            ):
                d2 = self._canon2d(d)
                if out_dim is None or int(d2.size(1)) == out_dim:
                    used.add(k)
                    return d2

        # shape fallback
        for k, d in deltas:
            if k in used:
                continue
            d2 = self._canon2d(d)
            if out_dim is None or int(d2.size(1)) == out_dim:
                used.add(k)
                return d2

        raise RuntimeError(
            f"FA: could not match any task output delta to output block '{block_name}'. "
            f"deltas={[(k, tuple(v.shape)) for k, v in deltas]}"
        )

    # -------------------------
    # main
    # -------------------------

    @torch.no_grad()
    def train_step(self, model, task, batch, device, state=None):
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, Mapping):
            raise NotImplementedError("FA expects tuple batch (x, y) in this minimal version.")

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)

        if not hasattr(model, "get_blocks"):
            raise NotImplementedError("FA expects model.get_blocks() (no fallback).")
        if "return_cache" not in model.forward.__code__.co_varnames:
            raise NotImplementedError("FA expects model(x, return_cache=True).")

        out, cache = model(x, return_cache=True)
        if not isinstance(cache, Mapping) or "block_inputs" not in cache:
            raise RuntimeError("FA expects cache['block_inputs'] from model(..., return_cache=True).")
        block_inputs: Mapping[str, Any] = cache["block_inputs"]

        # output deltas from task
        deltas_dict = self._task_output_deltas(task, out, y)
        out_deltas: List[Tuple[str, torch.Tensor]] = []
        for k, d in deltas_dict.items():
            d2 = self._canon2d(d)
            if d2.numel() > 0:
                out_deltas.append((k, d2))
        if not out_deltas:
            raise RuntimeError("FA: task.output_deltas returned no usable tensors.")

        # collect learnable blocks in forward order
        blocks = model.get_blocks()
        learnable_blocks = [
            b
            for b in blocks
            if isinstance(getattr(b, "module", None), (nn.Linear, nn.Conv2d))
        ]
        if not learnable_blocks:
            raise RuntimeError("FA: no (Linear|Conv2d) blocks found.")

        # group by chain, preserving order
        chains: Dict[str, List[Any]] = {}
        for b in learnable_blocks:
            name = str(getattr(b, "name", ""))
            cid = self._chain_id(name)
            chains.setdefault(cid, []).append(b)

        # run FA per chain (simple sequential assumption)
        for _, blks in chains.items():
            output_blocks = [b for b in blks if getattr(b, "is_output", False)]
            if not output_blocks:
                continue

            used = set()
            head_deltas: Dict[str, torch.Tensor] = {}
            for b in output_blocks:
                name = str(getattr(b, "name", ""))
                layer = b.module
                head_deltas[name] = self._match_head_delta(name, layer, out_deltas, used)

            # supervised: 1 head => on part du dernier output
            last_out = output_blocks[-1]
            last_name = str(getattr(last_out, "name", ""))
            delta_next = head_deltas[last_name]

            # grads for output layer
            if last_name not in block_inputs:
                raise RuntimeError(f"FA: missing cache['block_inputs'][{last_name}] for output block.")
            x_last = block_inputs[last_name]
            if not torch.is_tensor(x_last):
                raise RuntimeError(f"FA: output block '{last_name}' expects tensor input.")

            mod_last = last_out.module
            if isinstance(mod_last, nn.Linear):
                if x_last.dim() != 2:
                    raise RuntimeError(f"FA: output Linear '{last_name}' expects 2D input.")
                self._set_linear_grads(mod_last, x_last, delta_next)
            elif isinstance(mod_last, nn.Conv2d):
                z_last = mod_last(x_last)
                d4 = delta_next
                if d4.dim() == 2:
                    d4 = d4.view(d4.size(0), d4.size(1), 1, 1).expand_as(z_last)
                d4 = self._spatial_align(d4, (int(z_last.size(-2)), int(z_last.size(-1))))
                self._set_conv2d_grads(mod_last, x_last, d4)
                delta_next = d4
            else:
                raise RuntimeError(f"FA: unsupported output module type: {type(mod_last)}")

            # backward through remaining blocks
            idx_last = blks.index(last_out)
            for i in range(idx_last - 1, -1, -1):
                b = blks[i]
                name = str(getattr(b, "name", ""))
                mod = b.module

                b_next = blks[i + 1]
                name_next = str(getattr(b_next, "name", ""))
                mod_next = b_next.module

                if name not in block_inputs:
                    raise RuntimeError(f"FA: missing cache['block_inputs'][{name}] for hidden block.")
                x_i = block_inputs[name]
                if not torch.is_tensor(x_i):
                    raise RuntimeError(f"FA: hidden block '{name}' expects tensor input.")

                z_i = mod(x_i)

                self._ensure_B(name_next, mod_next, device=z_i.device, dtype=z_i.dtype)
                B_next = self._B[name_next]

                if isinstance(mod_next, nn.Linear):
                    delta_in = delta_next @ B_next  # [B, in_next]

                    if z_i.dim() == 2:
                        delta_i = delta_in * self._relu_deriv(z_i)
                        assert isinstance(mod, nn.Linear)
                        self._set_linear_grads(mod, x_i, delta_i)
                        delta_next = delta_i
                        continue

                    if z_i.dim() == 4:
                        H, W = int(z_i.size(-2)), int(z_i.size(-1))
                        delta4 = delta_in.view(delta_in.size(0), delta_in.size(1), 1, 1).expand(-1, -1, H, W)
                        if self.pool_backscale:
                            delta4 = delta4 / float(max(1, H * W))
                        delta_i = delta4 * self._relu_deriv(z_i)
                        assert isinstance(mod, nn.Conv2d)
                        self._set_conv2d_grads(mod, x_i, delta_i)
                        delta_next = delta_i
                        continue

                    raise RuntimeError(f"FA: unsupported activation dim {z_i.dim()} for block '{name}'.")

                elif isinstance(mod_next, nn.Conv2d):
                    if delta_next.dim() == 2:
                        if name_next not in block_inputs:
                            raise RuntimeError(f"FA: missing cache['block_inputs'][{name_next}] for next block.")
                        x_next = block_inputs[name_next]
                        z_next = mod_next(x_next)
                        delta_next = delta_next.view(delta_next.size(0), delta_next.size(1), 1, 1).expand_as(z_next)

                    if delta_next.dim() != 4:
                        raise RuntimeError(f"FA: expected 4D delta for Conv2d next block '{name_next}'.")

                    if name_next not in block_inputs:
                        raise RuntimeError(f"FA: missing cache['block_inputs'][{name_next}] for next block.")
                    x_next = block_inputs[name_next]
                    if not torch.is_tensor(x_next) or x_next.dim() != 4:
                        raise RuntimeError(f"FA: next Conv2d '{name_next}' expects 4D input.")

                    delta_in = torch.nn.grad.conv2d_input(
                        x_next.shape,
                        B_next,
                        delta_next,
                        stride=mod_next.stride,
                        padding=mod_next.padding,
                        dilation=mod_next.dilation,
                        groups=mod_next.groups,
                    )

                    if z_i.dim() != 4:
                        raise RuntimeError(f"FA: Conv2d mapping expects current z 4D, got {z_i.dim()}.")
                    delta_in = self._spatial_align(delta_in, (int(z_i.size(-2)), int(z_i.size(-1))))
                    delta_i = delta_in * self._relu_deriv(z_i)
                    assert isinstance(mod, nn.Conv2d)
                    self._set_conv2d_grads(mod, x_i, delta_i)
                    delta_next = delta_i
                    continue

                else:
                    raise RuntimeError(f"FA: unsupported next module type: {type(mod_next)}")

        params = list(model.parameters())
        self._clip_grads(params)
        self.optimizer.step()

        stats: Dict[str, float] = {}
        try:
            loss = task.loss(out, y)
            if torch.is_tensor(loss):
                stats["loss"] = float(loss.item())
            if hasattr(task, "metrics"):
                stats.update(task.metrics(out, y))
        except Exception:
            pass

        self.global_step += 1
        return stats
