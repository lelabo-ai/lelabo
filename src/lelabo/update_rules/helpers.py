"""Shared helper functions used by update rule implementations."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.activations import (
    LOCAL_RULE_SUPPORTED_ACTIVATIONS,
    local_rule_activation_derivative_from_preact,
    local_rule_activation_from_preact,
    local_rule_activation_name_from_value,
    normalize_local_rule_activation_name,
)

SUPPORTED_LOCAL_ACTIVATIONS = LOCAL_RULE_SUPPORTED_ACTIVATIONS


def normalize_activation_name(raw: str | None) -> str:
    return normalize_local_rule_activation_name(raw)


def resolve_activation_name(model) -> str:
    for attr in ("activation_name", "activation", "act_name"):
        if not hasattr(model, attr):
            continue
        raw = getattr(model, attr)
        resolved = local_rule_activation_name_from_value(raw)
        if resolved is not None:
            return resolved
    raise NotImplementedError(
        "Unable to resolve model activation. "
        "Set model.activation_name to one of: "
        + ", ".join(SUPPORTED_LOCAL_ACTIVATIONS)
        + "."
    )


def activation_derivative_from_preact(activation_name: str, preact: torch.Tensor) -> torch.Tensor:
    return local_rule_activation_derivative_from_preact(activation_name, preact)


def activation_from_preact(activation_name: str, preact: torch.Tensor) -> torch.Tensor:
    return local_rule_activation_from_preact(activation_name, preact)


def assign_param_grad_(param: torch.nn.Parameter, grad: torch.Tensor) -> None:
    g = grad.to(device=param.device, dtype=param.dtype)
    if param.grad is None:
        param.grad = g.clone()
    else:
        param.grad.copy_(g)


def assign_linear_grads_from_activations_(
    layer: nn.Linear,
    activations: torch.Tensor,
    delta: torch.Tensor,
    *,
    average_batch: bool = False,
    scale: float = 1.0,
) -> None:
    """Write Linear layer gradients from local activations and local delta.

    Args:
      activations: input of the linear layer, shape [B, in_features].
      delta: dL/du for this layer pre-activation u, shape [B, out_features].
      average_batch: divide gradients by B when True.
      scale: extra multiplicative factor applied to gradients.
    """
    if activations.dim() != 2 or delta.dim() != 2:
        raise ValueError("assign_linear_grads_from_activations_ expects 2D tensors.")
    if activations.size(0) != delta.size(0):
        raise ValueError("Batch size mismatch between activations and delta.")

    batch_size = float(max(1, activations.size(0)))
    d_weight = delta.transpose(0, 1) @ activations
    d_bias = delta.sum(dim=0) if layer.bias is not None else None

    if average_batch:
        d_weight = d_weight / batch_size
        if d_bias is not None:
            d_bias = d_bias / batch_size

    if scale != 1.0:
        d_weight = d_weight * float(scale)
        if d_bias is not None:
            d_bias = d_bias * float(scale)

    assign_param_grad_(layer.weight, d_weight)
    if layer.bias is not None and d_bias is not None:
        assign_param_grad_(layer.bias, d_bias)


def assign_conv2d_grads_from_activations_(
    layer: nn.Conv2d,
    activations: torch.Tensor,
    delta: torch.Tensor,
    *,
    average_batch: bool = False,
    scale: float = 1.0,
) -> None:
    """Write Conv2d layer gradients from local activations and local delta."""
    if activations.dim() != 4 or delta.dim() != 4:
        raise ValueError("assign_conv2d_grads_from_activations_ expects 4D tensors.")
    if activations.size(0) != delta.size(0):
        raise ValueError("Batch size mismatch between activations and delta.")
    if int(delta.size(1)) != int(layer.out_channels):
        raise ValueError(
            f"Delta channel mismatch for Conv2d: got {int(delta.size(1))}, "
            f"expected {int(layer.out_channels)}."
        )

    batch_size = float(max(1, activations.size(0)))
    d_weight = torch.nn.grad.conv2d_weight(
        activations,
        layer.weight.shape,
        delta,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    d_bias = delta.sum(dim=(0, 2, 3)) if layer.bias is not None else None

    if average_batch:
        d_weight = d_weight / batch_size
        if d_bias is not None:
            d_bias = d_bias / batch_size

    if scale != 1.0:
        d_weight = d_weight * float(scale)
        if d_bias is not None:
            d_bias = d_bias * float(scale)

    assign_param_grad_(layer.weight, d_weight)
    if layer.bias is not None and d_bias is not None:
        assign_param_grad_(layer.bias, d_bias)
"""Shared helper functions used by update rule implementations."""
