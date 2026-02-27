# lab/update_rules/helpers.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.batch import to_device


def set_grad_(param: torch.nn.Parameter, grad: torch.Tensor, scale: float = 1.0):
    g = grad.to(device=param.device, dtype=param.dtype)
    if scale != 1.0:
        g = g * float(scale)
    if param.grad is None:
        param.grad = g.clone()
    else:
        param.grad.copy_(g)


def set_linear_grads_(layer: nn.Linear, x: torch.Tensor, delta: torch.Tensor, scale: float = 1.0):
    B = x.size(0)
    dW = (delta.T @ x) / float(B)
    set_grad_(layer.weight, dW, scale=scale)
    if layer.bias is not None:
        db = delta.mean(dim=0)
        set_grad_(layer.bias, db, scale=scale)


def optimizer_step(optimizer: torch.optim.Optimizer, *, params_for_clip: Optional[List[torch.nn.Parameter]] = None, grad_clip: Optional[float] = None):
    if grad_clip is not None:
        if params_for_clip is None:
            raise ValueError("params_for_clip must be provided if grad_clip is not None")
        torch.nn.utils.clip_grad_norm_(params_for_clip, grad_clip)
    optimizer.step()


def ce_delta_logits(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, y_true.view(-1, 1), 1.0)
    return probs - onehot


def infer_activation_name(model, override: Optional[str] = None) -> str:
    if override is not None:
        return str(override).lower()
    if hasattr(model, "activation"):
        return str(model.activation).lower()
    return "relu"


def act_deriv_from_name(z: torch.Tensor, act_name: str, *, ste_heaviside: bool = False) -> torch.Tensor:
    act_name = str(act_name).lower()
    if act_name == "relu":
        return (z > 0).to(z.dtype)
    if act_name == "tanh":
        h = torch.tanh(z)
        return 1.0 - h * h
    if act_name == "sigmoid":
        h = torch.sigmoid(z)
        return h * (1.0 - h)
    if act_name == "heaviside":
        if ste_heaviside:
            return (z.abs() <= 1.0).to(z.dtype)
        return torch.zeros_like(z)
    return (z > 0).to(z.dtype)


@torch.no_grad()
def collect_linear_cache(model, x: torch.Tensor):
    if hasattr(model, "linears"):
        out_with_cache = None
        try:
            out_with_cache = model(x, return_cache=True)
        except TypeError as exc:
            if "return_cache" not in str(exc):
                raise
        if isinstance(out_with_cache, tuple) and len(out_with_cache) == 2:
            logits, cache = out_with_cache
            if (
                isinstance(cache, dict)
                and "block_inputs" in cache
                and "block_outputs" in cache
                and hasattr(model, "get_blocks")
            ):
                block_inputs = cache["block_inputs"]
                block_outputs = cache["block_outputs"]
                layer_cache: List[Tuple[torch.Tensor, torch.Tensor, nn.Linear]] = []

                for b in model.get_blocks():
                    name = getattr(b, "name", None)
                    module = getattr(b, "module", None)
                    if not isinstance(name, str) or not isinstance(module, nn.Linear):
                        continue
                    x_l = block_inputs.get(name, None) if isinstance(block_inputs, dict) else None
                    z_l = block_outputs.get(name, None) if isinstance(block_outputs, dict) else None
                    if torch.is_tensor(x_l) and torch.is_tensor(z_l):
                        layer_cache.append((x_l.detach(), z_l.detach(), module))

                if layer_cache:
                    return logits, layer_cache

    layer_cache: List[Tuple[torch.Tensor, torch.Tensor, nn.Linear]] = []
    hooks = []

    def hook_fn(module, inputs, output):
        x0 = inputs[0]
        z0 = output
        if x0.dim() == 2 and z0.dim() == 2:
            layer_cache.append((x0.detach(), z0.detach(), module))

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook_fn))

    logits = model(x)

    for h in hooks:
        h.remove()

    return logits, layer_cache


def split_hidden_and_output(layer_cache, logits: torch.Tensor):
    if len(layer_cache) == 0:
        return [], None

    out_idx = None
    for i in reversed(range(len(layer_cache))):
        if layer_cache[i][1].shape == logits.shape:
            out_idx = i
            break
    if out_idx is None:
        out_idx = len(layer_cache) - 1

    trimmed = layer_cache[: out_idx + 1]
    hidden = trimmed[:-1]
    out = trimmed[-1]
    return hidden, out


def ensure_fa_feedback(linear_layers: List[nn.Linear], feedback_scale: float, device, dtype, prev: Optional[List[torch.Tensor]]):
    L = len(linear_layers)
    if L < 2:
        return []

    if prev is not None and len(prev) == (L - 1):
        ok = True
        for l in range(L - 1):
            out_next = linear_layers[l + 1].out_features
            out_cur = linear_layers[l].out_features
            if prev[l].shape != (out_next, out_cur):
                ok = False
                break
        if ok:
            return [B.to(device=device, dtype=dtype) for B in prev]

    mats: List[torch.Tensor] = []
    for l in range(L - 1):
        out_next = linear_layers[l + 1].out_features
        out_cur = linear_layers[l].out_features
        scale = float(feedback_scale) / math.sqrt(max(out_next, 1))
        mats.append(torch.randn(out_next, out_cur, device=device, dtype=dtype) * scale)
    return mats


def ensure_dfa_feedback(hidden_layers: List[nn.Linear], out_dim: int, feedback_scale: float, device, dtype, prev: Optional[List[torch.Tensor]]):
    H = len(hidden_layers)
    if H <= 0:
        return []

    if prev is not None and len(prev) == H:
        ok = True
        for l, layer in enumerate(hidden_layers):
            if prev[l].shape != (out_dim, layer.out_features):
                ok = False
                break
        if ok:
            return [B.to(device=device, dtype=dtype) for B in prev]

    mats: List[torch.Tensor] = []
    scale = float(feedback_scale) / math.sqrt(max(out_dim, 1))
    for layer in hidden_layers:
        mats.append(torch.randn(out_dim, layer.out_features, device=device, dtype=dtype) * scale)
    return mats
