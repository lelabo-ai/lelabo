# lab/algorithms/update_rules/helpers.py
from __future__ import annotations

import math
from contextlib import contextmanager
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Batch / device utils
# -------------------------

def to_device(obj: Any, device: str):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(to_device(x, device) for x in obj)
    return obj


def is_hf_batch(batch: Any) -> bool:
    return isinstance(batch, Mapping)


def unpack_batch(batch: Any, device: str):
    """
    Returns:
      - mode: "hf" or "tuple"
      - payload:
          hf: (batch_dict,)
          tuple: (x, y) where y can be tensor or dict
    """
    if is_hf_batch(batch):
        b = to_device(batch, device)
        return "hf", (b,)
    x, y = batch
    x = to_device(x, device)
    y = to_device(y, device)
    return "tuple", (x, y)


# -------------------------
# Loss / stats standardisation
# -------------------------

def extract_loss_and_stats(res) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Accept:
      - loss_tensor
      - (loss_tensor, stats_dict)
      - {"loss": loss, ...scalars}
    """
    if torch.is_tensor(res):
        return res, {}
    if isinstance(res, tuple) and len(res) == 2:
        loss, stats = res
        if not torch.is_tensor(loss):
            raise TypeError("loss must be a torch.Tensor")
        out: Dict[str, float] = {}
        if isinstance(stats, Mapping):
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    out[k] = float(v)
        return loss, out
    if isinstance(res, Mapping):
        if "loss" not in res:
            raise ValueError("dict result must contain key 'loss'")
        loss = res["loss"]
        if not torch.is_tensor(loss):
            loss = torch.tensor(float(loss))
        out: Dict[str, float] = {}
        for k, v in res.items():
            if k == "loss":
                continue
            if isinstance(v, (int, float)):
                out[k] = float(v)
        return loss, out
    raise TypeError(f"Unsupported loss return type: {type(res)}")


# -------------------------
# Grad writing / optimizer step
# -------------------------

def set_grad_(param: torch.nn.Parameter, grad: torch.Tensor, scale: float = 1.0):
    g = grad.to(device=param.device, dtype=param.dtype)
    if scale != 1.0:
        g = g * float(scale)
    if param.grad is None:
        param.grad = g.clone()
    else:
        param.grad.copy_(g)


def set_linear_grads_(layer: nn.Linear, x: torch.Tensor, delta: torch.Tensor, scale: float = 1.0):
    """
    delta = dL/dz for that layer output (shape [B, out])
    x     = layer input (shape [B, in])
    writes grads into layer.weight.grad, layer.bias.grad
    """
    B = x.size(0)
    dW = (delta.T @ x) / float(B)
    set_grad_(layer.weight, dW, scale=scale)
    if layer.bias is not None:
        db = delta.mean(dim=0)
        set_grad_(layer.bias, db, scale=scale)


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    *,
    params_for_clip: Optional[List[torch.nn.Parameter]] = None,
    grad_clip: Optional[float] = None,
):
    if grad_clip is not None:
        if params_for_clip is None:
            raise ValueError("params_for_clip must be provided if grad_clip is not None")
        torch.nn.utils.clip_grad_norm_(params_for_clip, grad_clip)
    optimizer.step()


# -------------------------
# CE / activation derivatives
# -------------------------

def ce_delta_logits(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    dL/dlogits for CE-softmax.
    logits: [B,C], y_true: [B]
    """
    B = logits.size(0)
    probs = F.softmax(logits, dim=1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, y_true.view(-1, 1), 1.0)
    return (probs - onehot)


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
            return (z.abs() <= 1.0).to(z.dtype)  # simple STE band
        return torch.zeros_like(z)
    return (z > 0).to(z.dtype)


# -------------------------
# Cache collection for Linear layers (prefer return_cache, fallback hooks)
# -------------------------

def has_return_cache(model) -> bool:
    try:
        return "return_cache" in model.forward.__code__.co_varnames
    except Exception:
        return False


@torch.no_grad()
def collect_linear_cache(model, x: torch.Tensor):
    """
    Returns:
      logits, layer_cache where layer_cache is list of (x_l, z_l, layer_linear)
    Priority:
      - if model exposes model.linears and supports return_cache: use it (clean & stable)
      - else fallback to forward hooks on nn.Linear (2D only)
    """
    # Clean path: MLP-like (model.linears + return_cache)
    if hasattr(model, "linears") and has_return_cache(model):
        logits, cache = model(x, return_cache=True)
        linears = list(model.linears)
        layer_cache = []
        for i, layer in enumerate(linears):
            x_l = cache["inputs"][i]
            z_l = cache["preacts"][i]
            layer_cache.append((x_l.detach(), z_l.detach(), layer))
        return logits, layer_cache

    # Fallback path: hooks
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
    """
    For generic architectures, pick the last Linear whose output matches logits shape.
    Returns (hidden_cache, out_cache)
    """
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


# -------------------------
# Random feedback matrices init
# -------------------------

def ensure_fa_feedback(linear_layers: List[nn.Linear], feedback_scale: float, device, dtype, prev: Optional[List[torch.Tensor]]):
    """
    FA: B_l shape (out_{l+1}, out_l) for l=0..L-2
    """
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
    """
    DFA: B_l shape (C, out_l) for each hidden layer.
    """
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
