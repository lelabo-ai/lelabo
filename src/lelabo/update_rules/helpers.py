from __future__ import annotations

import torch
import torch.nn as nn


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
