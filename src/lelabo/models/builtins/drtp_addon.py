from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..registry import ModelContext, register_model


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _make_activation(name: str) -> tuple[str, nn.Module]:
    key = str(name).strip().lower()
    if key in {"none", "identity", "linear"}:
        return "identity", nn.Identity()
    if key == "relu":
        return "relu", nn.ReLU()
    if key == "tanh":
        return "tanh", nn.Tanh()
    if key == "sigmoid":
        return "sigmoid", nn.Sigmoid()
    raise ValueError(
        f"Unsupported activation '{name}'. Supported values: "
        "none/identity/linear, relu, tanh, sigmoid."
    )


@dataclass(frozen=True)
class _ConvLayout:
    out_channels: int
    kernel_size: int
    stride: int
    padding: int


@dataclass(frozen=True)
class _FCLayout:
    out_features: int


def _parse_topology(topology: str) -> list[tuple[str, _ConvLayout | _FCLayout]]:
    tokens = [tok.strip() for tok in str(topology).split("_") if tok.strip()]
    if not tokens:
        raise ValueError("Topology cannot be empty.")

    out: list[tuple[str, _ConvLayout | _FCLayout]] = []
    idx = 0
    while idx < len(tokens):
        tag = tokens[idx].upper()
        if tag == "CONV":
            if idx + 4 >= len(tokens):
                raise ValueError(
                    "Invalid CONV token format. Expected: CONV_{out_channels}_{kernel}_{stride}_{padding}."
                )
            try:
                layout = _ConvLayout(
                    out_channels=int(tokens[idx + 1]),
                    kernel_size=int(tokens[idx + 2]),
                    stride=int(tokens[idx + 3]),
                    padding=int(tokens[idx + 4]),
                )
            except ValueError as exc:
                raise ValueError(f"Invalid CONV numeric value near token index {idx}: {exc}") from exc
            if (
                layout.out_channels <= 0
                or layout.kernel_size <= 0
                or layout.stride <= 0
                or layout.padding < 0
            ):
                raise ValueError(
                    "CONV values must satisfy: out_channels>0, kernel_size>0, stride>0, padding>=0."
                )
            out.append(("CONV", layout))
            idx += 5
            continue

        if tag == "FC":
            if idx + 1 >= len(tokens):
                raise ValueError("Invalid FC token format. Expected: FC_{out_features}.")
            try:
                layout = _FCLayout(out_features=int(tokens[idx + 1]))
            except ValueError as exc:
                raise ValueError(f"Invalid FC numeric value near token index {idx}: {exc}") from exc
            if layout.out_features <= 0:
                raise ValueError("FC out_features must be > 0.")
            out.append(("FC", layout))
            idx += 2
            continue

        raise ValueError(
            f"Unknown topology token '{tokens[idx]}'. Expected tokens starting with CONV or FC."
        )

    if not out:
        raise ValueError("Topology parsing produced no layer.")
    return out


class _LayerRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str):
        key = str(name).strip().upper()

        def _decorator(builder: Callable[..., nn.Module]):
            if key in self._builders:
                raise ValueError(f"[drtp_addon.local_registry] '{name}' already registered.")
            self._builders[key] = builder
            return builder

        return _decorator

    def build(self, name: str, *args, **kwargs) -> nn.Module:
        key = str(name).strip().upper()
        if key not in self._builders:
            raise ValueError(
                f"[drtp_addon.local_registry] Unknown layer type '{name}'. "
                f"Available: {sorted(self._builders)}."
            )
        return self._builders[key](*args, **kwargs)


class _ConvStage(nn.Module):
    def __init__(self, *, in_channels: int, layout: _ConvLayout, activation: str) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=int(layout.out_channels),
            kernel_size=int(layout.kernel_size),
            stride=int(layout.stride),
            padding=int(layout.padding),
            bias=True,
        )
        self.activation_name, self.activation = _make_activation(activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preact = self.conv(x)
        out = self.activation(preact)
        out = self.pool(out)
        return out, preact


class _FCStage(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        layout: _FCLayout,
        activation: str,
        dropout: float,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.linear = nn.Linear(int(in_features), int(layout.out_features), bias=True)
        if bool(zero_init):
            nn.init.zeros_(self.linear.weight)
        self.activation_name, self.activation = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.drop(x)
        preact = self.linear(x)
        out = self.activation(preact)
        return out, preact


LOCAL_LAYER_REGISTRY = _LayerRegistry()


@LOCAL_LAYER_REGISTRY.register("CONV")
def _build_conv_stage(*, in_channels: int, layout: _ConvLayout, activation: str) -> _ConvStage:
    return _ConvStage(in_channels=in_channels, layout=layout, activation=activation)


@LOCAL_LAYER_REGISTRY.register("FC")
def _build_fc_stage(
    *,
    in_features: int,
    layout: _FCLayout,
    activation: str,
    dropout: float,
    zero_init: bool,
) -> _FCStage:
    return _FCStage(
        in_features=in_features,
        layout=layout,
        activation=activation,
        dropout=dropout,
        zero_init=zero_init,
    )


def _conv2d_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return ((int(size) - int(kernel) + (2 * int(padding))) // int(stride)) + 1


def _pool2d_out(size: int) -> int:
    return int(size) // 2


class DRTPAddonNet(nn.Module):
    """
    Self-contained DRTP-style network:
    - topology parser from a single string
    - local per-file registry for layer builders (LOCAL_LAYER_REGISTRY)
    - standard cache contract support for local update-rules
    """

    def __init__(
        self,
        *,
        topology: str,
        num_classes: int,
        input_shape: Sequence[int] | None,
        in_dim: int | None,
        conv_activation: str = "tanh",
        hidden_activation: str = "tanh",
        output_activation: str = "identity",
        activation_name: str | None = None,
        dropout: float = 0.0,
        fc_zero_init: bool = False,
        align_output_to_num_classes: bool = True,
    ) -> None:
        super().__init__()
        self.topology = str(topology)
        self.num_classes = int(num_classes)
        self.conv_activation = str(conv_activation)
        self.hidden_activation = str(hidden_activation)
        self.output_activation = str(output_activation)
        self.activation_name, _ = _make_activation(
            str(activation_name if activation_name is not None else hidden_activation)
        )
        self.dropout = float(dropout)
        self.fc_zero_init = bool(fc_zero_init)
        self.align_output_to_num_classes = bool(align_output_to_num_classes)

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        parsed = _parse_topology(self.topology)
        if parsed[-1][0] != "FC":
            raise ValueError("Topology must end with an FC output layer.")

        if self.align_output_to_num_classes:
            _, last_layout = parsed[-1]
            assert isinstance(last_layout, _FCLayout)
            if int(last_layout.out_features) != self.num_classes:
                parsed = list(parsed)
                parsed[-1] = ("FC", _FCLayout(out_features=self.num_classes))

        img_shape = tuple(int(v) for v in input_shape) if input_shape is not None else None
        if img_shape is not None and len(img_shape) not in {1, 3}:
            raise ValueError(f"input_shape must be len=1 or len=3 when provided, got {img_shape}.")

        current_channels: int | None = None
        current_h: int | None = None
        current_w: int | None = None
        current_flat: int | None = int(in_dim) if in_dim is not None else None

        if img_shape is not None:
            if len(img_shape) == 3:
                current_channels, current_h, current_w = img_shape
                if current_flat is None:
                    current_flat = int(current_channels * current_h * current_w)
            else:
                if current_flat is None:
                    current_flat = int(img_shape[0])

        if current_flat is None:
            raise ValueError(
                "Unable to infer model input dimension. Provide ctx.in_dim or ctx.input_shape."
            )

        self.layers = nn.ModuleList()
        self._flatten_before_stage: set[int] = set()

        for idx, (kind, layout) in enumerate(parsed):
            is_output = idx == (len(parsed) - 1)

            if kind == "CONV":
                assert isinstance(layout, _ConvLayout)
                if current_channels is None or current_h is None or current_w is None:
                    raise ValueError(
                        "CONV layer requires image-like input_shape=(C,H,W) in ModelContext."
                    )
                stage = LOCAL_LAYER_REGISTRY.build(
                    "CONV",
                    in_channels=current_channels,
                    layout=layout,
                    activation=self.conv_activation,
                )
                self.layers.append(stage)

                current_channels = int(layout.out_channels)
                current_h = _conv2d_out(current_h, layout.kernel_size, layout.stride, layout.padding)
                current_w = _conv2d_out(current_w, layout.kernel_size, layout.stride, layout.padding)
                current_h = _pool2d_out(current_h)
                current_w = _pool2d_out(current_w)
                if current_h <= 0 or current_w <= 0:
                    raise ValueError(
                        f"Topology produced invalid spatial shape after layer index {idx}: "
                        f"(H={current_h}, W={current_w})."
                    )
                current_flat = int(current_channels * current_h * current_w)
                continue

            assert kind == "FC"
            assert isinstance(layout, _FCLayout)
            if current_flat is None:
                raise ValueError("Cannot infer in_features for FC layer.")

            if idx > 0 and parsed[idx - 1][0] == "CONV":
                self._flatten_before_stage.add(len(self.layers))

            stage_activation = self.output_activation if is_output else self.hidden_activation
            stage = LOCAL_LAYER_REGISTRY.build(
                "FC",
                in_features=current_flat,
                layout=layout,
                activation=stage_activation,
                dropout=self.dropout,
                zero_init=self.fc_zero_init,
            )
            self.layers.append(stage)
            current_flat = int(layout.out_features)

    def forward(self, x: torch.Tensor):
        for idx, stage in enumerate(self.layers):
            if idx in self._flatten_before_stage:
                x = x.reshape(x.size(0), -1)
            x, _ = stage(x)
        return x


def _pick(args, params: Mapping[str, Any], names: Sequence[str], default: Any) -> Any:
    for name in names:
        if name in params:
            return params[name]
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value
    return default


@register_model("drtp_addon")
@register_model("drtp_topology")
def build_drtp_addon(ctx: ModelContext, args):
    model_params = getattr(args, "model_params", None)
    params = dict(model_params) if isinstance(model_params, Mapping) else {}

    default_topology = f"CONV_32_5_1_2_FC_1000_FC_{int(ctx.num_classes)}"
    topology = str(_pick(args, params, ("topology",), default_topology))
    conv_act = str(_pick(args, params, ("conv_act", "conv_activation"), "tanh"))
    hidden_act = str(_pick(args, params, ("hidden_act", "hidden_activation", "activation"), "tanh"))
    output_act = str(_pick(args, params, ("output_act", "output_activation"), "identity"))
    activation_name = _pick(args, params, ("activation_name",), None)
    if activation_name is not None:
        activation_name = str(activation_name)
    dropout = float(_pick(args, params, ("dropout",), 0.0))
    fc_zero_init = _as_bool(_pick(args, params, ("fc_zero_init",), False))
    align_output = _as_bool(
        _pick(
            args,
            params,
            ("align_output_to_num_classes", "force_output_dim"),
            True,
        )
    )

    return DRTPAddonNet(
        topology=topology,
        num_classes=int(ctx.num_classes),
        input_shape=ctx.input_shape,
        in_dim=ctx.in_dim,
        conv_activation=conv_act,
        hidden_activation=hidden_act,
        output_activation=output_act,
        activation_name=activation_name,
        dropout=dropout,
        fc_zero_init=fc_zero_init,
        align_output_to_num_classes=align_output,
    )


__all__ = ["DRTPAddonNet", "LOCAL_LAYER_REGISTRY", "build_drtp_addon"]
