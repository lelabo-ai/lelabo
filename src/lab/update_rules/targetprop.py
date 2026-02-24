# lab/update_rules/targetprop.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule
from .helpers import to_device, set_grad_
from ..core.steps import metric_payload_from_outputs


class TargetPropagation(UpdateRule):
    """
    Target Propagation, block-based, ReLU-only, supports:
      - Supervised: (x, y_true) with y_true = labels tensor
      - RL (PPO/DQN-style): (x, y_dict) with task.output_deltas(out, y_dict)

    Model requirements:
      - model.get_blocks() returning blocks with: name, module, is_output
      - model(x, return_cache=True) returning (out, cache)
      - cache["block_inputs"][block_name] must exist for each block name

    Update rule:
      - Train inverse maps (decoders) g_l to reconstruct previous block input from next activation.
      - Build output targets:
          t_out = out_concat - beta * delta_out_concat
        where delta_out_concat comes from:
          - supervised CE (if labels) OR
          - task.output_deltas(out, y_dict) (RL)
      - Propagate targets backward through decoders:
          t_prev = h_prev + g(t_next) - g(h_next)
      - Update forward weights locally:
          For hidden blocks: delta = (t_next - h_next) * relu'(z)
          For output blocks: delta = (-beta * delta_out)   (local head update)
    """

    def __init__(
        self,
        fwd_lr: float = 0.05,
        inv_lr: float = 0.05,
        beta: float = 1.0,
        noise_std: float = 0.1,
        eps: float = 1e-8,
        fwd_optimizer: Optional[torch.optim.Optimizer] = None,
        inv_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__()
        self.fwd_lr = float(fwd_lr)
        self.inv_lr = float(inv_lr)
        self.beta = float(beta)
        self.noise_std = float(noise_std)
        self.eps = float(eps)
        self.fwd_optimizer = fwd_optimizer
        self.inv_optimizer = inv_optimizer

        # decoders for hidden transitions + last hidden <- output_concat
        self.decoders: Optional[nn.ModuleList] = None
        self._dec_sig: Optional[List[Tuple[int, int]]] = None  # list of (in_dim, out_dim) for Linear(out_dim -> in_dim)

    # ============================================================
    # Basics
    # ============================================================

    def _relu_deriv(self, z: torch.Tensor) -> torch.Tensor:
        return (z > 0).to(z.dtype)

    def _ce_delta_logits(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # dL/dlogits for CE: (softmax - onehot) / B
        B = logits.size(0)
        p = F.softmax(logits, dim=1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, y_true.view(-1, 1), 1.0)
        return (p - onehot) / float(B)

    # ============================================================
    # Collect linear blocks
    # ============================================================

    def _linear_blocks(self, model) -> Tuple[List[Any], List[Any]]:
        if not hasattr(model, "get_blocks"):
            raise NotImplementedError("TargetPropagation expects model.get_blocks().")
        blocks = model.get_blocks()
        linear_blocks = [b for b in blocks if isinstance(getattr(b, "module", None), nn.Linear)]
        hidden = [b for b in linear_blocks if not getattr(b, "is_output", False)]
        output = [b for b in linear_blocks if getattr(b, "is_output", False)]
        if not output:
            raise RuntimeError("TargetPropagation: no output blocks (is_output=True).")
        if not hidden:
            # You *can* run with no hidden (just a head), but TP becomes pointless; still allow.
            hidden = []
        return hidden, output

    # ============================================================
    # Output deltas (supervised or RL)
    # ============================================================

    def _task_output_deltas(self, task, out: Any, y: Any) -> Dict[str, torch.Tensor]:
        # RL path: delegate to task.output_deltas
        if isinstance(y, Mapping):
            if not hasattr(task, "output_deltas"):
                raise NotImplementedError("TargetPropagation RL requires task.output_deltas(out, y).")
            d = task.output_deltas(out, y)
            if not isinstance(d, Mapping):
                raise RuntimeError("task.output_deltas must return a dict[str, Tensor].")
            return {k: v for k, v in d.items() if torch.is_tensor(v)}

        # supervised classification: out is logits tensor
        if torch.is_tensor(out) and torch.is_tensor(y) and y.dim() == 1:
            return {"logits": self._ce_delta_logits(out, y.long())}

        raise NotImplementedError(
            "TargetPropagation: unsupported (out,y). Provide labels tensor for supervised or dict y for RL with task.output_deltas."
        )

    def _concat_outputs_and_deltas(self, out: Any, deltas: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, int]]]:
        """
        Build out_concat and delta_concat (both [B, C_total]).
        Also returns a layout list of (key, width) to split back for head updates.
        """
        parts_out: List[torch.Tensor] = []
        parts_delta: List[torch.Tensor] = []
        layout: List[Tuple[str, int]] = []

        # common keys ordering for actor-critic
        keys = []
        if isinstance(out, Mapping):
            # stable order
            for k in ("logits", "value", "q_values"):
                if k in out and k in deltas:
                    keys.append(k)
            # any remaining keys
            for k in deltas.keys():
                if k not in keys and (isinstance(out, Mapping) and k in out):
                    keys.append(k)
        else:
            # supervised logits tensor
            keys = ["logits"]

        for k in keys:
            o = out[k] if isinstance(out, Mapping) else out
            d = deltas[k]
            o2 = o if o.dim() == 2 else o.view(o.size(0), -1)
            d2 = d if d.dim() == 2 else d.view(d.size(0), -1)
            if o2.size(1) != d2.size(1):
                raise RuntimeError(f"TargetPropagation: delta shape mismatch for '{k}': out={tuple(o2.shape)}, delta={tuple(d2.shape)}")
            parts_out.append(o2)
            parts_delta.append(d2)
            layout.append((k, int(o2.size(1))))

        if not parts_out:
            raise RuntimeError("TargetPropagation: could not build output concat (no matching out/delta keys).")

        out_cat = torch.cat(parts_out, dim=1)
        delta_cat = torch.cat(parts_delta, dim=1)
        return out_cat, delta_cat, layout

    # ============================================================
    # Decoders
    # ============================================================

    def _ensure_decoders(self, dims: List[Tuple[int, int]], device, dtype):
        """
        dims: list of (in_dim, out_dim) for each decoder Linear(out_dim -> in_dim)
        """
        if self.decoders is not None and self._dec_sig == dims:
            for d in self.decoders:
                d.to(device=device, dtype=dtype)
            return

        self.decoders = nn.ModuleList()
        self._dec_sig = list(dims)

        for (in_dim, out_dim) in dims:
            dec = nn.Linear(out_dim, in_dim, bias=True).to(device=device, dtype=dtype)
            nn.init.normal_(dec.weight, mean=0.0, std=0.01)
            nn.init.zeros_(dec.bias)
            self.decoders.append(dec)

        if self.inv_optimizer is not None:
            self.inv_optimizer.param_groups = []
            self.inv_optimizer.add_param_group({"params": list(self.decoders.parameters())})

    def _train_inverse(self, h_list: List[torch.Tensor], out_cat: torch.Tensor):
        """
        Train decoders for:
          - hidden l: h_l  ~ dec_l(h_{l+1})
          - last hidden: h_last ~ dec_last(out_cat)
        """
        assert self.decoders is not None
        B = h_list[0].size(0)

        if self.inv_optimizer is not None:
            self.inv_optimizer.zero_grad(set_to_none=True)

        Lh = len(h_list)
        # decoders[0..Lh-2] for hidden transitions, last decoder for out->last_hidden
        for l in range(Lh):
            if l == Lh - 1:
                y_in = out_cat
                target = h_list[-1]
                dec = self.decoders[-1]
            else:
                y_in = h_list[l + 1]
                target = h_list[l]
                dec = self.decoders[l]

            if self.noise_std > 0.0:
                y_in = y_in + self.noise_std * torch.randn_like(y_in)

            recon = dec(y_in)
            err = (target - recon)

            dW = (err.T @ y_in) / float(B)
            db = err.mean(dim=0)

            if self.inv_optimizer is None:
                dec.weight.add_(self.inv_lr * dW)
                dec.bias.add_(self.inv_lr * db)
            else:
                set_grad_(dec.weight, -dW)
                set_grad_(dec.bias, -db)

        if self.inv_optimizer is not None:
            self.inv_optimizer.step()

    # ============================================================
    # Targets propagation
    # ============================================================

    def _propagate_targets(self, h_list: List[torch.Tensor], out_cat: torch.Tensor, t_out: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns targets for each hidden activation h_list[i].
        """
        assert self.decoders is not None
        Lh = len(h_list)
        targets: List[Optional[torch.Tensor]] = [None] * Lh

        t_next = t_out
        h_next = out_cat

        for i in reversed(range(Lh)):
            # decoder index: i==last hidden => use last decoder, else use decoder i
            dec = self.decoders[-1] if i == (Lh - 1) else self.decoders[i]
            h_prev = h_list[i]

            t_prev = h_prev + dec(t_next) - dec(h_next)
            targets[i] = t_prev

            # update "next" for next iteration
            t_next = t_prev
            h_next = h_prev

        return [t for t in targets if t is not None]  # type: ignore

    # ============================================================
    # Forward updates
    # ============================================================

    def _apply_linear_update(self, layer: nn.Linear, x: torch.Tensor, delta: torch.Tensor):
        B = x.size(0)
        dW = (delta.T @ x) / float(B)
        db = delta.mean(dim=0)

        if self.fwd_optimizer is None:
            layer.weight.add_(self.fwd_lr * dW)
            if layer.bias is not None:
                layer.bias.add_(self.fwd_lr * db)
        else:
            set_grad_(layer.weight, -dW)
            if layer.bias is not None:
                set_grad_(layer.bias, -db)

    # ============================================================
    # Main
    # ============================================================

    @torch.no_grad()
    def train_step(self, model, task, batch, device, state=None):
        model.train()

        if isinstance(batch, Mapping):
            raise NotImplementedError("TargetPropagation: batch must be tuple (x, y).")

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)

        try:
            out_cache = model(x, return_cache=True)
        except TypeError as exc:
            if "return_cache" in str(exc):
                raise NotImplementedError("TargetPropagation expects model(x, return_cache=True).") from exc
            raise
        if not (isinstance(out_cache, tuple) and len(out_cache) == 2):
            raise RuntimeError("TargetPropagation expects model(x, return_cache=True) to return (out, cache).")
        out, cache = out_cache
        if not isinstance(cache, Mapping) or "block_inputs" not in cache:
            raise RuntimeError("TargetPropagation expects cache['block_inputs'] from the model.")
        block_inputs: Mapping[str, Any] = cache["block_inputs"]

        hidden_blocks, output_blocks = self._linear_blocks(model)

        # Build hidden activations list h_list = outputs of hidden blocks: h = layer(x)
        h_list: List[torch.Tensor] = []
        for b in hidden_blocks:
            name = getattr(b, "name", None)
            layer: nn.Linear = b.module  # type: ignore
            if name is None or name not in block_inputs:
                raise RuntimeError(f"TargetPropagation: missing block_inputs for hidden block '{name}'.")
            x_l = block_inputs[name]
            if not torch.is_tensor(x_l) or x_l.dim() != 2:
                raise RuntimeError(f"TargetPropagation: block '{name}' expects 2D input, got {type(x_l)} shape={getattr(x_l, 'shape', None)}.")
            h_list.append(layer(x_l))  # pre-activation output (Linear output)

        # Output deltas from task
        deltas = self._task_output_deltas(task, out, y)

        # Build output concat and delta concat
        out_cat, delta_cat, layout = self._concat_outputs_and_deltas(out, deltas)

        # If no hidden blocks, just do local head update and return
        if len(hidden_blocks) == 0:
            if self.fwd_optimizer is not None:
                self.fwd_optimizer.zero_grad(set_to_none=True)

            # Update output blocks by matching shapes via layout order
            offset = 0
            for b in output_blocks:
                name = getattr(b, "name", None)
                layer: nn.Linear = b.module  # type: ignore
                if name is None or name not in block_inputs:
                    continue
                x_o = block_inputs[name]
                if not torch.is_tensor(x_o) or x_o.dim() != 2:
                    continue

                # choose a slice that matches this head out_features
                chosen = None
                for (k, w) in layout:
                    if int(layer.out_features) == int(w):
                        chosen = delta_cat[:, offset:offset + w]
                        break
                    offset += w
                if chosen is None:
                    continue

                # local head update: delta = -beta * dL/dout
                self._apply_linear_update(layer, x_o, -self.beta * chosen)

            if self.fwd_optimizer is not None:
                self.fwd_optimizer.step()

            self.global_step += 1
            return {}

        # Ensure decoders: (hidden i) reconstruct prev hidden from next hidden, and last hidden from out_cat
        # dims for decoders:
        #   for i=0..Lh-2: in_dim = hidden_i_dim, out_dim = hidden_{i+1}_dim
        #   last: in_dim = hidden_last_dim, out_dim = out_cat_dim
        device0 = h_list[0].device
        dtype0 = h_list[0].dtype
        dims: List[Tuple[int, int]] = []
        for i in range(len(h_list) - 1):
            dims.append((int(h_list[i].size(1)), int(h_list[i + 1].size(1))))
        dims.append((int(h_list[-1].size(1)), int(out_cat.size(1))))
        self._ensure_decoders(dims, device=device0, dtype=dtype0)

        # Train inverse maps
        self._train_inverse(h_list, out_cat)

        # Output target
        t_out = out_cat - self.beta * delta_cat

        # Propagate targets to hidden activations
        targets_h = self._propagate_targets(h_list, out_cat, t_out)

        # Update forward weights (hidden blocks)
        if self.fwd_optimizer is not None:
            self.fwd_optimizer.zero_grad(set_to_none=True)

        for i, b in enumerate(hidden_blocks):
            name = b.name
            layer: nn.Linear = b.module  # type: ignore
            x_l = block_inputs[name]
            h_next = h_list[i]         # Linear output
            t_next = targets_h[i]
            z = h_next                 # preact for ReLU deriv
            delta = (t_next - h_next) * self._relu_deriv(z)
            self._apply_linear_update(layer, x_l, delta)

        # Update output heads (local)
        # We try to match per-key deltas to output blocks by name hints + shape.
        # We form head delta as: delta_head = -beta * dL/dout_head
        # where dout_head is logits/value/q_values.
        out_deltas_2d: List[Tuple[str, torch.Tensor]] = []
        for k, d in deltas.items():
            d2 = d if d.dim() == 2 else d.view(d.size(0), -1)
            out_deltas_2d.append((k, d2))

        used = set()
        for b in output_blocks:
            name = str(getattr(b, "name", ""))
            layer: nn.Linear = b.module  # type: ignore
            if name not in block_inputs:
                continue
            x_o = block_inputs[name]
            if not torch.is_tensor(x_o) or x_o.dim() != 2:
                continue

            chosen: Optional[torch.Tensor] = None
            lname = name.lower()

            # name-hint first (actor/critic)
            for k, d in out_deltas_2d:
                if k in used:
                    continue
                if ("logits" in k and "actor" in lname) or ("value" in k and "critic" in lname) or ("q_values" in k and ("q" in lname or "head" in lname)):
                    if int(layer.out_features) == int(d.size(1)):
                        chosen = d
                        used.add(k)
                        break

            # shape fallback
            if chosen is None:
                for k, d in out_deltas_2d:
                    if k in used:
                        continue
                    if int(layer.out_features) == int(d.size(1)):
                        chosen = d
                        used.add(k)
                        break

            if chosen is None:
                continue

            self._apply_linear_update(layer, x_o, -self.beta * chosen)

        if self.fwd_optimizer is not None:
            self.fwd_optimizer.step()

        # stats (best-effort)
        stats: Dict[str, Any] = {}
        try:
            loss = task.loss(out, y) if isinstance(y, Mapping) else task.loss(out, y)
            if torch.is_tensor(loss):
                stats["loss"] = float(loss.item())
            if hasattr(task, "metrics"):
                met = task.metrics(out, y)
                if isinstance(met, Mapping):
                    for key, value in met.items():
                        if isinstance(value, (int, float)):
                            stats[str(key)] = float(value)
                        elif str(key).startswith("__metric_"):
                            stats[str(key)] = value
            if isinstance(y, torch.Tensor):
                stats.update(metric_payload_from_outputs(out, y))
        except Exception:
            pass

        self.global_step += 1
        return stats
