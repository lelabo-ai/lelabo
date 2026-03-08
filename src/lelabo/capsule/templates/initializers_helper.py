from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from lelabo.initializers.registry import InitializerContext


_CONV_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def parse_seed(raw: Any) -> int | None:
    if raw is None:
        return None
    return int(raw)


def module_devices(model: nn.Module) -> list[int]:
    device_ids: list[int] = []
    for p in model.parameters():
        if p.is_cuda and p.device.index is not None:
            idx = int(p.device.index)
            if idx not in device_ids:
                device_ids.append(idx)
    return device_ids


@contextmanager
def maybe_local_seed(model: nn.Module, seed: int | None):
    if seed is None:
        yield
        return

    devices = module_devices(model)
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def bias_mode(params: dict[str, Any]) -> str:
    mode = str(params.get("bias", "zeros")).strip().lower()
    if mode not in {"zeros", "none"}:
        raise ValueError("initializer.params.bias must be one of: zeros, none.")
    return mode


def norm_modes(params: dict[str, Any]) -> tuple[str, str]:
    weight_mode = str(params.get("norm_weight", "ones")).strip().lower()
    bias_mode_ = str(params.get("norm_bias", "zeros")).strip().lower()

    if weight_mode not in {"ones", "zeros", "none"}:
        raise ValueError("initializer.params.norm_weight must be one of: ones, zeros, none.")
    if bias_mode_ not in {"ones", "zeros", "none"}:
        raise ValueError("initializer.params.norm_bias must be one of: ones, zeros, none.")

    return weight_mode, bias_mode_


def apply_norm_init(model: nn.Module, *, weight_mode: str, bias_mode: str) -> None:
    for module in model.modules():
        if not isinstance(module, _NORM_TYPES):
            continue

        if getattr(module, "weight", None) is not None and torch.is_tensor(module.weight):
            if weight_mode == "ones":
                nn.init.ones_(module.weight)
            elif weight_mode == "zeros":
                nn.init.zeros_(module.weight)

        if getattr(module, "bias", None) is not None and torch.is_tensor(module.bias):
            if bias_mode == "ones":
                nn.init.ones_(module.bias)
            elif bias_mode == "zeros":
                nn.init.zeros_(module.bias)


def apply_affine_init(
    model: nn.Module,
    *,
    weight_init: Callable[[torch.Tensor], None],
    bias_mode: str,
) -> None:
    for module in model.modules():
        if not isinstance(module, (nn.Linear, *_CONV_TYPES)):
            continue

        if getattr(module, "weight", None) is not None and torch.is_tensor(module.weight):
            weight_init(module.weight)

        if bias_mode == "zeros" and getattr(module, "bias", None) is not None and torch.is_tensor(module.bias):
            nn.init.zeros_(module.bias)


def make_initializer(
    ctx: InitializerContext,
    *,
    weight_init: Callable[[torch.Tensor], None],
    name: str = "custom",
    extra_log: Callable[[], dict[str, Any]] | None = None,
) -> Callable[[nn.Module], dict[str, Any]]:
    """
    Build a LeLabo initializer callable from a tensor-level `weight_init` function.

    Supported generic params in `[initializer.params]`:
      - bias: "zeros" | "none"
      - norm_weight: "ones" | "zeros" | "none"
      - norm_bias: "ones" | "zeros" | "none"
      - seed: optional integer

    The returned callable:
      - applies `weight_init` to Linear/Conv weights
      - optionally zeros affine biases
      - optionally initializes normalization layers
      - optionally uses a local RNG seed
    """
    params = ctx.initializer_params()
    seed = parse_seed(params.get("seed"))
    affine_bias_mode = bias_mode(params)
    norm_weight_mode, norm_bias_mode = norm_modes(params)

    @torch.no_grad()
    def _apply(model: nn.Module) -> dict[str, Any]:
        with maybe_local_seed(model, seed):
            apply_affine_init(model, weight_init=weight_init, bias_mode=affine_bias_mode)
            apply_norm_init(model, weight_mode=norm_weight_mode, bias_mode=norm_bias_mode)

        out: dict[str, Any] = {
            "initializer": name,
            "seed": seed,
            "bias": affine_bias_mode,
            "norm_weight": norm_weight_mode,
            "norm_bias": norm_bias_mode,
        }
        if extra_log is not None:
            out.update(extra_log())
        return out

    return _apply
