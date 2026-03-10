from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Activation modules accepted by cache auto-pairing logic.
CACHE_AUTO_PAIR_ACTIVATION_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.PReLU,
    nn.ELU,
    nn.CELU,
    nn.SELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
    nn.Tanh,
    nn.Sigmoid,
    nn.Hardtanh,
    nn.Hardsigmoid,
    nn.Softplus,
    nn.Softsign,
)

LOCAL_RULE_SUPPORTED_ACTIVATIONS: tuple[str, ...] = (
    "relu",
    "relu6",
    "tanh",
    "sigmoid",
    "gelu",
    "silu",
    "mish",
    "softsign",
    "hardsigmoid",
    "selu",
    "identity",
)

_LOCAL_RULE_MODULE_TO_NAME: tuple[tuple[type[nn.Module], str], ...] = (
    (nn.ReLU, "relu"),
    (nn.ReLU6, "relu6"),
    (nn.Tanh, "tanh"),
    (nn.Sigmoid, "sigmoid"),
    (nn.GELU, "gelu"),
    (nn.SiLU, "silu"),
    (nn.Mish, "mish"),
    (nn.Softsign, "softsign"),
    (nn.Hardsigmoid, "hardsigmoid"),
    (nn.SELU, "selu"),
    (nn.Identity, "identity"),
)


def normalize_local_rule_activation_name(raw: str | None) -> str:
    act = str(raw).strip().lower()
    if act not in LOCAL_RULE_SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{raw}'. "
            f"Supported: {', '.join(LOCAL_RULE_SUPPORTED_ACTIVATIONS)}."
        )
    return act


def local_rule_activation_name_from_value(raw: Any) -> str | None:
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in LOCAL_RULE_SUPPORTED_ACTIVATIONS:
            return token
        return None

    for module_type, name in _LOCAL_RULE_MODULE_TO_NAME:
        if isinstance(raw, module_type):
            return name

    fn_name = getattr(raw, "__name__", None)
    if isinstance(fn_name, str):
        token = fn_name.strip().lower()
        if token in LOCAL_RULE_SUPPORTED_ACTIVATIONS:
            return token

    return None


def local_rule_activation_from_preact(activation_name: str, preact: torch.Tensor) -> torch.Tensor:
    act = normalize_local_rule_activation_name(activation_name)
    if act == "relu":
        return torch.relu(preact)
    if act == "relu6":
        return torch.clamp(preact, min=0.0, max=6.0)
    if act == "tanh":
        return torch.tanh(preact)
    if act == "sigmoid":
        return torch.sigmoid(preact)
    if act == "gelu":
        return F.gelu(preact)
    if act == "silu":
        return F.silu(preact)
    if act == "mish":
        return F.mish(preact)
    if act == "softsign":
        return F.softsign(preact)
    if act == "hardsigmoid":
        return F.hardsigmoid(preact)
    if act == "selu":
        return F.selu(preact)
    return preact


def local_rule_activation_derivative_from_preact(activation_name: str, preact: torch.Tensor) -> torch.Tensor:
    act = normalize_local_rule_activation_name(activation_name)
    if act == "relu":
        return (preact > 0).to(preact.dtype)
    if act == "relu6":
        return ((preact > 0) & (preact < 6)).to(preact.dtype)
    if act == "tanh":
        t = torch.tanh(preact)
        return 1.0 - t * t
    if act == "sigmoid":
        s = torch.sigmoid(preact)
        return s * (1.0 - s)
    if act == "gelu":
        cdf = 0.5 * (1.0 + torch.erf(preact / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * preact * preact) / math.sqrt(2.0 * math.pi)
        return cdf + preact * pdf
    if act == "silu":
        s = torch.sigmoid(preact)
        return s + preact * s * (1.0 - s)
    if act == "mish":
        sp = F.softplus(preact)
        t = torch.tanh(sp)
        return t + preact * torch.sigmoid(preact) * (1.0 - t * t)
    if act == "softsign":
        denom = 1.0 + torch.abs(preact)
        return 1.0 / (denom * denom)
    if act == "hardsigmoid":
        return ((preact > -3.0) & (preact < 3.0)).to(preact.dtype) / 6.0
    if act == "selu":
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        positive = (preact > 0).to(preact.dtype) * scale
        negative = (preact <= 0).to(preact.dtype) * (scale * alpha) * torch.exp(preact)
        return positive + negative
    return torch.ones_like(preact)
