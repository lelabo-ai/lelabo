# lab/update_rules/dfa.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .base import UpdateRule
from ..core.batch import to_device


class DirectFeedbackAlignment(UpdateRule):
    """
    DFA (no autograd), block-based, ReLU-only, minimal, supervised + RL via task.output_deltas(), no fallback.

    Requirements:
      - model.get_blocks() -> list of blocks with fields: name, module, is_output
      - model(x, return_cache=True) -> (out, cache)
      - cache["block_inputs"][block_name] gives 2D input to that Linear

    For RL:
      - task.output_deltas(out, y_dict) must exist and return dict[str, Tensor]
        e.g. PPO: {"logits": [B,A], "value": [B] or [B,1]}
             DQN: {"q_values": [B,A]}

    DFA rule (per chain):
      - output deltas come from the task
      - for each chain (actor / critic / default):
          * build delta_out_cat for that chain (usually one head)
          * hidden blocks: delta_l = (delta_out_cat @ B_l) * relu'(z_l)
          * output blocks: delta_head = task delta for that head
      - write grads, optimizer.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        feedback_scale: float = 1.0,
        grad_clip: Optional[float] = None,
        grad_scale: float = 1.0,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.feedback_scale = float(feedback_scale)
        self.grad_clip = grad_clip
        self.grad_scale = float(grad_scale)

        # feedback per chain: dict[chain_id] -> list[Tensor]
        self._B_by_chain: Dict[str, List[torch.Tensor]] = {}
        self._Bsig_by_chain: Dict[str, Tuple[Tuple[int, int], ...]] = {}

    # -------------------------
    # small utils
    # -------------------------

    def _relu_deriv(self, z: torch.Tensor) -> torch.Tensor:
        return (z > 0).to(z.dtype)

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

    def _clip_grads(self, params: Sequence[nn.Parameter]):
        if self.grad_clip is None:
            return
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(self.grad_clip))

    def _ensure_feedback_chain(self, chain_id: str, hidden_linears: Sequence[nn.Linear], C_total: int, device, dtype):
        sig = tuple((int(C_total), int(l.out_features)) for l in hidden_linears)
        if chain_id in self._B_by_chain and self._Bsig_by_chain.get(chain_id, None) == sig:
            self._B_by_chain[chain_id] = [B.to(device=device, dtype=dtype) for B in self._B_by_chain[chain_id]]
            return

        mats: List[torch.Tensor] = []
        for (_C, out_l) in sig:
            B = torch.empty(C_total, out_l, device=device, dtype=dtype)
            torch.nn.init.normal_(B, mean=0.0, std=0.05)
            B.mul_(self.feedback_scale)
            mats.append(B)

        self._B_by_chain[chain_id] = mats
        self._Bsig_by_chain[chain_id] = sig

    # -------------------------
    # output deltas
    # -------------------------

    def _task_output_deltas(self, task, out: Any, y: Any) -> Dict[str, torch.Tensor]:
        if not hasattr(task, "output_deltas"):
            raise NotImplementedError("DFA requires task.output_deltas(out, y).")
        d = task.output_deltas(out, y)
        if not isinstance(d, Mapping):
            raise RuntimeError("task.output_deltas must return dict[str, Tensor].")
        return {k: v for k, v in d.items() if torch.is_tensor(v)}

    def _canon2d(self, t: torch.Tensor) -> torch.Tensor:
        return t if t.dim() == 2 else t.view(t.size(0), -1)

    def _chain_id(self, block_name: str) -> str:
        n = block_name
        if n.startswith("actor."):
            return "actor"
        if n.startswith("critic."):
            return "critic"
        return "default"

    def _match_head_delta(self, block_name: str, layer: nn.Linear, deltas: List[Tuple[str, torch.Tensor]], used: set) -> torch.Tensor:
        """
        Choose a delta tensor for this output head using name hints then shape.
        """
        lname = block_name.lower()

        # name-hints
        for k, d in deltas:
            if k in used:
                continue
            if ("logits" in k and ("actor" in lname or "logits" in lname)) or ("value" in k and ("critic" in lname or "value" in lname)) or ("q_values" in k and ("q" in lname or "head" in lname)):
                if int(layer.out_features) == int(d.size(1)):
                    used.add(k)
                    return d

        # shape fallback
        for k, d in deltas:
            if k in used:
                continue
            if int(layer.out_features) == int(d.size(1)):
                used.add(k)
                return d

        raise RuntimeError(
            f"DFA: could not match any task output delta to output block '{block_name}' "
            f"(out_features={layer.out_features}). deltas={[(k, tuple(d.shape)) for k, d in deltas]}"
        )

    # -------------------------
    # main
    # -------------------------

    @torch.no_grad()
    def train_step(self, model, task, batch, device, state=None):
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, Mapping):
            raise NotImplementedError("DFA expects tuple batch (x, y). No HF dict path in this minimal version.")

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)

        if not hasattr(model, "get_blocks"):
            raise NotImplementedError("DFA expects model.get_blocks() (no fallback).")
        try:
            out_cache = model(x, return_cache=True)
        except TypeError as exc:
            if "return_cache" in str(exc):
                raise NotImplementedError("DFA expects model(x, return_cache=True).") from exc
            raise
        if not (isinstance(out_cache, tuple) and len(out_cache) == 2):
            raise RuntimeError("DFA expects model(x, return_cache=True) to return (out, cache).")
        out, cache = out_cache
        if not isinstance(cache, Mapping) or "block_inputs" not in cache:
            raise RuntimeError("DFA expects cache['block_inputs'] from model(..., return_cache=True).")
        block_inputs: Mapping[str, Any] = cache["block_inputs"]

        # output deltas from task
        deltas_dict = self._task_output_deltas(task, out, y)

        out_deltas: List[Tuple[str, torch.Tensor]] = []
        for k, d in deltas_dict.items():
            d2 = self._canon2d(d)
            if d2.numel() > 0:
                out_deltas.append((k, d2))
        if not out_deltas:
            raise RuntimeError("DFA: task.output_deltas returned no usable tensors.")

        # collect Linear blocks
        blocks = model.get_blocks()
        linear_blocks = [b for b in blocks if isinstance(getattr(b, "module", None), nn.Linear)]
        if not linear_blocks:
            raise RuntimeError("DFA: no Linear blocks found (this implementation is Linear-only).")

        # group by chain
        chains: Dict[str, List[Any]] = {}
        for b in linear_blocks:
            name = str(getattr(b, "name", ""))
            cid = self._chain_id(name)
            chains.setdefault(cid, []).append(b)

        # run DFA per chain
        for cid, blks in chains.items():
            hidden_blocks = [b for b in blks if not getattr(b, "is_output", False)]
            output_blocks = [b for b in blks if getattr(b, "is_output", False)]
            if not output_blocks:
                # nothing to learn in this chain
                continue

            # --- output deltas for this chain (usually 1 head) ---
            used = set()
            head_deltas: Dict[str, torch.Tensor] = {}
            for b in output_blocks:
                name = str(getattr(b, "name", ""))
                layer: nn.Linear = b.module  # type: ignore
                d = self._match_head_delta(name, layer, out_deltas, used)
                head_deltas[name] = d  # dL/d(out_head)

            # concat chain output deltas for hidden DFA
            delta_out_cat = torch.cat([head_deltas[str(getattr(b, "name", ""))] for b in output_blocks], dim=1)
            C_total = int(delta_out_cat.size(1))

            # feedback matrices for hidden blocks in this chain
            hidden_linears = [b.module for b in hidden_blocks]  # type: ignore
            self._ensure_feedback_chain(cid, hidden_linears, C_total=C_total, device=delta_out_cat.device, dtype=delta_out_cat.dtype)
            Bmats = self._B_by_chain[cid]

            # --- hidden grads ---
            for idx, b in enumerate(hidden_blocks):
                name = str(getattr(b, "name", ""))
                layer: nn.Linear = b.module  # type: ignore
                if name not in block_inputs:
                    raise RuntimeError(f"DFA: missing cache['block_inputs'][{name}] for hidden block.")
                x_l = block_inputs[name]
                if not torch.is_tensor(x_l) or x_l.dim() != 2:
                    raise RuntimeError(f"DFA: hidden block '{name}' expects 2D input tensor.")

                z_l = layer(x_l)
                Bmat = Bmats[idx]  # [C_total, out_l]
                delta_l = (delta_out_cat @ Bmat) * self._relu_deriv(z_l)
                self._set_linear_grads(layer, x_l, delta_l)

            # --- output grads ---
            for b in output_blocks:
                name = str(getattr(b, "name", ""))
                layer: nn.Linear = b.module  # type: ignore
                if name not in block_inputs:
                    raise RuntimeError(f"DFA: missing cache['block_inputs'][{name}] for output block.")
                x_o = block_inputs[name]
                if not torch.is_tensor(x_o) or x_o.dim() != 2:
                    raise RuntimeError(f"DFA: output block '{name}' expects 2D input tensor.")
                self._set_linear_grads(layer, x_o, head_deltas[name])

        # optimizer step
        params = list(model.parameters())
        self._clip_grads(params)
        self.optimizer.step()

        # --------- logging (best effort) ---------
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
