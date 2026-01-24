# lab/algorithms/update_rules/feedback_alignment.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import UpdateRule
from ...core.batch import to_device
from ...core.steps import maybe_accuracy_from_logits


class FeedbackAlignment(UpdateRule):
    """
    Feedback Alignment (FA), block-based, ReLU-only, minimal, no fallback.

    Works for:
      - Supervised classification: y is label tensor, out is logits tensor
      - RL (PPO/DQN/...): y is dict and task.output_deltas(out, y) exists

    Requirements:
      - model.get_blocks() -> BlockSpec-like objects with fields: name, module, is_output
      - model(x, return_cache=True) -> (out, cache)
      - cache["block_inputs"] provides 2D input tensor for each Linear block name

    Algorithm (FA):
      - compute output deltas (task-provided or CE)
      - map output deltas into deltas for output blocks
      - propagate hidden deltas backwards using fixed random feedback matrices B_l:
          delta_l = (delta_{l+1} @ B_l) * relu'(z_l)
      - write .grad for each Linear weight/bias, optimizer.step()
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

        # One B per hidden transition:
        #   for l in [0..L-2], B_l shape [out_{l+1}, out_l]
        self.B_mats: Optional[List[torch.Tensor]] = None
        self._B_sig: Optional[Tuple[Tuple[int, int], ...]] = None

    # ============================================================
    # ReLU derivative
    # ============================================================

    def _relu_deriv(self, z: torch.Tensor) -> torch.Tensor:
        return (z > 0).to(z.dtype)

    # ============================================================
    # Supervised CE delta
    # ============================================================

    def _ce_delta_logits(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        B = logits.size(0)
        p = F.softmax(logits, dim=1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, y_true.view(-1, 1), 1.0)
        return (p - onehot) / float(B)

    # ============================================================
    # Feedback init / reuse
    # ============================================================

    def _ensure_feedback(self, linears: Sequence[nn.Linear], device, dtype):
        if len(linears) < 2:
            self.B_mats = []
            self._B_sig = tuple()
            return

        sig = tuple((int(linears[l + 1].out_features), int(linears[l].out_features)) for l in range(len(linears) - 1))
        if self.B_mats is not None and self._B_sig == sig:
            self.B_mats = [B.to(device=device, dtype=dtype) for B in self.B_mats]
            return

        mats: List[torch.Tensor] = []
        for (out_next, out_l) in sig:
            B = torch.empty(out_next, out_l, device=device, dtype=dtype)
            torch.nn.init.normal_(B, mean=0.0, std=0.05)
            B.mul_(self.feedback_scale)
            mats.append(B)

        self.B_mats = mats
        self._B_sig = sig

    # ============================================================
    # Output deltas (supervised or RL via task.output_deltas)
    # ============================================================

    def _task_output_deltas(self, task, out: Any, y: Any) -> Dict[str, torch.Tensor]:
        # RL path (y is dict)
        if isinstance(y, Mapping):
            if not hasattr(task, "output_deltas"):
                raise NotImplementedError("FeedbackAlignment RL requires task.output_deltas(out, y).")
            d = task.output_deltas(out, y)
            if not isinstance(d, Mapping):
                raise RuntimeError("task.output_deltas must return dict[str, Tensor].")
            return {k: v for k, v in d.items() if torch.is_tensor(v)}

        # supervised classification path: out is logits tensor, y is labels
        if torch.is_tensor(out) and torch.is_tensor(y) and y.dim() == 1:
            return {"logits": self._ce_delta_logits(out, y.long())}

        raise NotImplementedError(
            "FeedbackAlignment: unsupported (out,y). Provide labels tensor for supervised or dict y for RL with task.output_deltas."
        )

    # ============================================================
    # Helpers: write grads for Linear
    # ============================================================

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

    # ============================================================
    # Main
    # ============================================================

    @torch.no_grad()
    def train_step(self, model, task, batch, device):
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, Mapping):
            raise NotImplementedError("FeedbackAlignment: batch must be tuple (x, y).")

        x, y = batch
        x = to_device(x, device)
        y = to_device(y, device)

        if not hasattr(model, "get_blocks"):
            raise NotImplementedError("FeedbackAlignment expects model.get_blocks() (no fallback).")
        if "return_cache" not in model.forward.__code__.co_varnames:
            raise NotImplementedError("FeedbackAlignment expects model(x, return_cache=True).")

        out, cache = model(x, return_cache=True)
        if not isinstance(cache, Mapping) or "block_inputs" not in cache:
            raise RuntimeError("FeedbackAlignment expects cache['block_inputs'] from model(..., return_cache=True).")
        block_inputs: Mapping[str, Any] = cache["block_inputs"]

        # Collect Linear blocks
        blocks = model.get_blocks()
        linear_blocks = [b for b in blocks if isinstance(getattr(b, "module", None), nn.Linear)]
        if not linear_blocks:
            raise RuntimeError("FeedbackAlignment supports only Linear-block models (MLP-style).")

        # Split hidden/output
        hidden_blocks = [b for b in linear_blocks if not getattr(b, "is_output", False)]
        output_blocks = [b for b in linear_blocks if getattr(b, "is_output", False)]
        if not output_blocks:
            raise RuntimeError("FeedbackAlignment: no output blocks (is_output=True).")

        # Linear list in global order (for feedback shapes)
        linears = [b.module for b in linear_blocks]  # type: ignore

        # Feedback matrices between consecutive linears
        # NOTE: This assumes linear_blocks are ordered from input->...->output in get_blocks().
        self._ensure_feedback(linears, device=x.device, dtype=x.dtype)

        # ------------------------------------------------
        # 1) Output deltas
        # ------------------------------------------------
        deltas_dict = self._task_output_deltas(task, out, y)

        # Convert to 2D
        out_deltas_2d: List[Tuple[str, torch.Tensor]] = []
        for k, d in deltas_dict.items():
            d2 = d if d.dim() == 2 else d.view(d.size(0), -1)
            out_deltas_2d.append((k, d2))

        # Assign a delta to each output block (by name hint, then by shape)
        output_deltas_by_name: Dict[str, torch.Tensor] = {}
        used_keys = set()

        for b in output_blocks:
            name = str(getattr(b, "name", ""))
            layer: nn.Linear = b.module  # type: ignore

            chosen: Optional[torch.Tensor] = None
            lname = name.lower()

            # name-hint first
            for k, d in out_deltas_2d:
                if k in used_keys:
                    continue
                if ("logits" in k and "actor" in lname) or ("value" in k and "critic" in lname) or ("q_values" in k and ("q" in lname or "head" in lname)):
                    if int(layer.out_features) == int(d.size(1)):
                        chosen = d
                        used_keys.add(k)
                        break

            # shape fallback
            if chosen is None:
                for k, d in out_deltas_2d:
                    if k in used_keys:
                        continue
                    if int(layer.out_features) == int(d.size(1)):
                        chosen = d
                        used_keys.add(k)
                        break

            if chosen is None:
                raise RuntimeError(
                    f"FeedbackAlignment: could not match any task output delta to output block '{name}' "
                    f"(out_features={layer.out_features}). task deltas={[(k, tuple(d.shape)) for k, d in out_deltas_2d]}"
                )

            output_deltas_by_name[name] = chosen  # dL/d(out_head)

        # ------------------------------------------------
        # 2) Propagate deltas backward (FA)
        # ------------------------------------------------
        # We'll compute deltas for each linear in *linear_blocks order*.
        # For output layers, we take the matched delta (dL/dout) directly.
        # For hidden layers, FA recursion uses B matrices.

        # Build a per-linear delta list aligned with linear_blocks
        L = len(linear_blocks)
        deltas: List[Optional[torch.Tensor]] = [None] * L

        # Set deltas for output blocks
        for i, b in enumerate(linear_blocks):
            if getattr(b, "is_output", False):
                name = str(getattr(b, "name", ""))
                deltas[i] = output_deltas_by_name[name]

        # Backward pass: find next delta that is already set (works for multiple heads at end, e.g. actor+critic)
        # Strategy:
        #   - walk from end to start
        #   - maintain "delta_next" as concatenation if multiple consecutive output blocks exist
        #
        # But standard PPO has 2 heads at the end (actor.head, critic.head). They are distinct linears, not stacked.
        # For FA layer-by-layer, we'd need a single delta_{l+1}. The simplest consistent way:
        #   - if there are multiple output blocks, we define delta_next_cat = concat(deltas for all output blocks at that depth)
        #   - and feedback matrix from that cat space to previous layer must match.
        #
        # HOWEVER, the feedback matrices were initialized using consecutive linears, which doesn't model branching.
        # So for multi-head models, we restrict FA to the chain order returned by get_blocks().
        # That means: your ActorCriticDiscrete.get_blocks() should list actor blocks THEN critic blocks,
        # which is two separate chains. We'll run FA independently on each chain (recommended).
        #
        # => Implementation below detects separate chains by group prefix in names:
        #    "actor." and "critic." (or "q." etc.) and runs FA per chain.
        #
        # This keeps FA "true" and avoids incorrect mixing across heads.

        # ------------------------------------------------
        # 3) Run FA per chain (actor / critic / default)
        # ------------------------------------------------
        def chain_id(block_name: str) -> str:
            n = block_name
            if n.startswith("actor."):
                return "actor"
            if n.startswith("critic."):
                return "critic"
            return "default"

        # group indices by chain preserving order
        chains: Dict[str, List[int]] = {}
        for i, b in enumerate(linear_blocks):
            bid = chain_id(str(getattr(b, "name", "")))
            chains.setdefault(bid, []).append(i)

        # Ensure feedback matrices per chain (we create B per chain, not global)
        # We'll store them in a dict keyed by chain id for stability.
        if not hasattr(self, "_B_by_chain"):
            self._B_by_chain = {}  # type: ignore

        for cid, idxs in chains.items():
            chain_linears = [linear_blocks[i].module for i in idxs]  # type: ignore
            if len(chain_linears) < 2:
                continue

            # signature per chain
            sig = tuple((int(chain_linears[j + 1].out_features), int(chain_linears[j].out_features)) for j in range(len(chain_linears) - 1))
            B_list = self._B_by_chain.get(cid, None)  # type: ignore

            if B_list is None or getattr(self, "_Bsig_by_chain", {}).get(cid, None) != sig:  # type: ignore
                mats: List[torch.Tensor] = []
                for (out_next, out_l) in sig:
                    B = torch.empty(out_next, out_l, device=x.device, dtype=x.dtype)
                    torch.nn.init.normal_(B, mean=0.0, std=0.05)
                    B.mul_(self.feedback_scale)
                    mats.append(B)
                self._B_by_chain[cid] = mats  # type: ignore
                if not hasattr(self, "_Bsig_by_chain"):
                    self._Bsig_by_chain = {}  # type: ignore
                self._Bsig_by_chain[cid] = sig  # type: ignore
            else:
                # move to device/dtype
                self._B_by_chain[cid] = [B.to(device=x.device, dtype=x.dtype) for B in B_list]  # type: ignore

            # propagate within chain
            Bmats = self._B_by_chain[cid]  # type: ignore

            # find output index in this chain (last is assumed output; if multiple, we just require the last one)
            # NOTE: for MLPClassifier chain, there's exactly one head.
            # For actor/critic, each chain has exactly one head.
            # If chain has multiple outputs, you'd need a more explicit design.
            local_deltas: List[Optional[torch.Tensor]] = [None] * len(idxs)
            for j, gi in enumerate(idxs):
                if getattr(linear_blocks[gi], "is_output", False):
                    local_deltas[j] = deltas[gi]

            # last delta must exist
            if local_deltas[-1] is None:
                # maybe output isn't last (weird), find the last set
                last_set = None
                for j in reversed(range(len(local_deltas))):
                    if local_deltas[j] is not None:
                        last_set = j
                        break
                if last_set is None:
                    continue
                # treat it as output for propagation
                # (still expects a proper linear chain though)

            # backward
            for j in reversed(range(len(idxs) - 1)):
                if local_deltas[j + 1] is None:
                    continue
                gi = idxs[j]
                x_j = block_inputs[str(getattr(linear_blocks[gi], "name", ""))]
                layer_j: nn.Linear = linear_blocks[gi].module  # type: ignore
                z_j = layer_j(x_j)

                delta_next = local_deltas[j + 1]
                Bmat = Bmats[j]  # [out_{j+1}, out_j]
                local_deltas[j] = (delta_next @ Bmat) * self._relu_deriv(z_j)

            # write back
            for j, gi in enumerate(idxs):
                if local_deltas[j] is not None:
                    deltas[gi] = local_deltas[j]

        # ------------------------------------------------
        # 4) Write grads for each linear block
        # ------------------------------------------------
        for i, b in enumerate(linear_blocks):
            name = str(getattr(b, "name", ""))
            layer: nn.Linear = b.module  # type: ignore
            if name not in block_inputs:
                raise RuntimeError(f"FeedbackAlignment: missing cache['block_inputs'][{name}].")
            x_l = block_inputs[name]
            if not torch.is_tensor(x_l) or x_l.dim() != 2:
                raise RuntimeError(f"FeedbackAlignment: block '{name}' expects 2D input tensor.")
            if deltas[i] is None:
                # if not computed (e.g., chain mismatch), skip rather than write garbage
                continue
            self._set_linear_grads(layer, x_l, deltas[i])

        # Step
        params = list(model.parameters())
        self._clip_grads(params)
        self.optimizer.step()

        # Stats (best-effort)
        stats: Dict[str, float] = {}
        try:
            if torch.is_tensor(out):
                # supervised
                loss = task.loss(out, y)
                stats["loss"] = float(loss.item())
                stats["acc"] = maybe_accuracy_from_logits(out, y)
            elif isinstance(out, Mapping):
                # RL
                loss = task.loss(out, y)
                if torch.is_tensor(loss):
                    stats["loss"] = float(loss.item())
                if "logits" in out and not isinstance(y, Mapping):
                    stats["acc"] = maybe_accuracy_from_logits(out["logits"], y)
                if hasattr(task, "metrics"):
                    stats.update(task.metrics(out, y))
        except Exception:
            pass

        self.global_step += 1
        return stats
