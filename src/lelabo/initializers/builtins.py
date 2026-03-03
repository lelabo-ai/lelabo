from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from .registry import InitializerContext, register_initializer


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


def _parse_seed(raw: Any) -> int | None:
    if raw is None:
        return None
    return int(raw)


def _module_devices(model: nn.Module) -> list[int]:
    device_ids: list[int] = []
    for p in model.parameters():
        if p.is_cuda and p.device.index is not None:
            idx = int(p.device.index)
            if idx not in device_ids:
                device_ids.append(idx)
    return device_ids


@contextmanager
def _maybe_local_seed(model: nn.Module, seed: int | None):
    if seed is None:
        yield
        return
    devices = _module_devices(model)
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def _bias_mode(params: dict[str, Any]) -> str:
    mode = str(params.get("bias", "zeros")).strip().lower()
    if mode not in {"zeros", "none"}:
        raise ValueError("initializer.params.bias must be one of: zeros, none.")
    return mode


def _norm_modes(params: dict[str, Any]) -> tuple[str, str]:
    w = str(params.get("norm_weight", "ones")).strip().lower()
    b = str(params.get("norm_bias", "zeros")).strip().lower()
    if w not in {"ones", "zeros", "none"}:
        raise ValueError("initializer.params.norm_weight must be one of: ones, zeros, none.")
    if b not in {"ones", "zeros", "none"}:
        raise ValueError("initializer.params.norm_bias must be one of: ones, zeros, none.")
    return w, b


def _apply_norm_init(model: nn.Module, *, weight_mode: str, bias_mode: str) -> None:
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


def _apply_affine_init(
    model: nn.Module,
    *,
    weight_init: Callable[[torch.Tensor], None],
    bias_mode: str,
) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Linear, *_CONV_TYPES)):
            if getattr(module, "weight", None) is not None and torch.is_tensor(module.weight):
                weight_init(module.weight)
            if bias_mode == "zeros" and getattr(module, "bias", None) is not None and torch.is_tensor(module.bias):
                nn.init.zeros_(module.bias)


def _make_init_callable(
    ctx: InitializerContext,
    *,
    weight_init: Callable[[torch.Tensor], None],
) -> Callable[[nn.Module], dict[str, Any]]:
    params = ctx.initializer_params()
    seed = _parse_seed(params.get("seed"))
    bias_mode = _bias_mode(params)
    norm_w, norm_b = _norm_modes(params)

    @torch.no_grad()
    def _apply(model: nn.Module) -> dict[str, Any]:
        with _maybe_local_seed(model, seed):
            _apply_affine_init(model, weight_init=weight_init, bias_mode=bias_mode)
            _apply_norm_init(model, weight_mode=norm_w, bias_mode=norm_b)
        return {
            "initializer": "custom",
            "seed": seed,
            "bias": bias_mode,
            "norm_weight": norm_w,
            "norm_bias": norm_b,
        }

    return _apply


@register_initializer("torch_default")
def build_none(_ctx: InitializerContext):
    @torch.no_grad()
    def _apply(_model: nn.Module) -> dict[str, Any]:
        return {"initializer": "none"}

    return _apply


@register_initializer("kaiming_uniform")
def build_kaiming_uniform(ctx: InitializerContext):
    params = ctx.initializer_params()
    a = float(params.get("a", 0.0))
    mode = str(params.get("mode", "fan_in"))
    nonlinearity = str(params.get("nonlinearity", "relu"))

    def _weight(w: torch.Tensor) -> None:
        nn.init.kaiming_uniform_(w, a=a, mode=mode, nonlinearity=nonlinearity)

    return _make_init_callable(ctx, weight_init=_weight)


@register_initializer("kaiming_normal")
def build_kaiming_normal(ctx: InitializerContext):
    params = ctx.initializer_params()
    a = float(params.get("a", 0.0))
    mode = str(params.get("mode", "fan_in"))
    nonlinearity = str(params.get("nonlinearity", "relu"))

    def _weight(w: torch.Tensor) -> None:
        nn.init.kaiming_normal_(w, a=a, mode=mode, nonlinearity=nonlinearity)

    return _make_init_callable(ctx, weight_init=_weight)


@register_initializer("xavier_uniform")
def build_xavier_uniform(ctx: InitializerContext):
    params = ctx.initializer_params()
    gain = float(params.get("gain", 1.0))

    def _weight(w: torch.Tensor) -> None:
        nn.init.xavier_uniform_(w, gain=gain)

    return _make_init_callable(ctx, weight_init=_weight)


@register_initializer("xavier_normal")
def build_xavier_normal(ctx: InitializerContext):
    params = ctx.initializer_params()
    gain = float(params.get("gain", 1.0))

    def _weight(w: torch.Tensor) -> None:
        nn.init.xavier_normal_(w, gain=gain)

    return _make_init_callable(ctx, weight_init=_weight)


@register_initializer("orthogonal")
def build_orthogonal(ctx: InitializerContext):
    params = ctx.initializer_params()
    gain = float(params.get("gain", 1.0))

    def _weight(w: torch.Tensor) -> None:
        if w.dim() < 2:
            nn.init.normal_(w, mean=0.0, std=0.01)
            return
        nn.init.orthogonal_(w, gain=gain)

    return _make_init_callable(ctx, weight_init=_weight)
