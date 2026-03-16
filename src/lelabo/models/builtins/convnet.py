# lab/models/convnet.py
"""Convolutional network baselines for vision tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from ..registry import ModelContext, register_model


def _make_activation(name: str):
    act = str(name).strip().lower()
    if act in {"identity", "none", "linear"}:
        return "identity", nn.Identity()
    if act == "relu":
        return "relu", nn.ReLU(inplace=False)
    if act == "tanh":
        return "tanh", nn.Tanh()
    if act == "sigmoid":
        return "sigmoid", nn.Sigmoid()
    raise ValueError(
        f"Unsupported activation '{name}'. "
        "Supported values: identity, relu, tanh, sigmoid."
    )


@register_model("cnn")
def build_cnn(ctx: ModelContext, args):
    if ctx.in_channels is None:
        raise ValueError("CNN needs ctx.in_channels")

    model_params = getattr(args, "model_params", None)
    params = dict(model_params) if isinstance(model_params, Mapping) else {}

    def _pick(name: str, default):
        if name in params:
            return params[name]
        value = getattr(args, name, default)
        return default if value is None else value

    hidden_activation = _pick("activation", _pick("conv_activation", _pick("conv_act", "relu")))
    output_activation = _pick("output_activation", _pick("head_activation", "identity"))

    return ConvNetClassifier(
        in_channels=ctx.in_channels,
        num_classes=ctx.num_classes,
        channels=_pick("channels", None),
        depth=_pick("depth", 3),
        base_width=_pick("base_width", 32),
        width_factor=_pick("width_factor", 2),
        kernel_sizes=_pick("kernel_sizes", 3),
        use_bn=_pick("use_bn", True),
        pool_every=_pick("pool_every", 1),
        pools=_pick("pools", None),
        pool_kernel=_pick("pool_kernel", 2),
        activation=hidden_activation,
        output_activation=output_activation,
        head_mode=_pick("head_mode", "flatten"),
        input_shape=ctx.input_shape,
    )


class ClassicCNN(nn.Module):
    """Classic CNN: (Conv -> BN -> Act -> Pool) x N -> (GAP|Flatten) -> Linear."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        widths: Sequence[int],
        kernel_sizes: Sequence[int],
        pools: Sequence[bool],
        use_bn: bool,
        pool_kernel: int,
        activation: str = "relu",
        output_activation: str = "identity",
        head_mode: str = "gap",
        input_shape: Sequence[int] | None = None,
    ):
        super().__init__()
        self.activation_name, _ = _make_activation(activation)
        self.output_activation_name, self.output_activation = _make_activation(output_activation)
        self.head_mode = str(head_mode).strip().lower()
        self.input_shape = tuple(int(v) for v in input_shape) if input_shape is not None else None
        if not widths:
            raise ValueError("ClassicCNN requires at least one conv width.")
        if not (len(widths) == len(kernel_sizes) == len(pools)):
            raise ValueError("widths, kernel_sizes, pools must share the same length.")
        if self.head_mode not in {"gap", "flatten"}:
            raise ValueError("head_mode must be one of: gap, flatten.")

        self._conv_names: list[str] = []
        self._act_names: list[str] = []
        c_in = int(in_channels)
        for i, (w, k, do_pool) in enumerate(zip(widths, kernel_sizes, pools), start=1):
            conv_name = f"conv{i}"
            bn_name = f"bn{i}"
            act_name = f"act{i}"
            pool_name = f"pool{i}"

            setattr(
                self,
                conv_name,
                nn.Conv2d(
                    c_in,
                    int(w),
                    kernel_size=int(k),
                    stride=1,
                    padding=int(k) // 2,
                    bias=not bool(use_bn),
                ),
            )
            setattr(self, bn_name, nn.BatchNorm2d(int(w)) if bool(use_bn) else nn.Identity())
            _, act_mod = _make_activation(self.activation_name)
            setattr(self, act_name, act_mod)
            setattr(self, pool_name, nn.MaxPool2d(int(pool_kernel)) if bool(do_pool) else nn.Identity())

            self._conv_names.append(conv_name)
            self._act_names.append(act_name)
            c_in = int(w)

        if self.head_mode == "gap":
            self.gap: nn.Module | None = nn.AdaptiveAvgPool2d((1, 1))
            head_in_features = int(c_in)
        else:
            self.gap = None
            if self.input_shape is None:
                raise ValueError("head_mode='flatten' requires input_shape=(C,H,W).")
            if len(self.input_shape) != 3:
                raise ValueError("input_shape must have 3 values: (C,H,W).")
            in_c, in_h, in_w = self.input_shape
            if in_c != int(in_channels):
                raise ValueError(
                    f"input_shape channel mismatch: got C={in_c}, expected {int(in_channels)}."
                )
            if in_h <= 0 or in_w <= 0:
                raise ValueError("input_shape spatial dimensions must be > 0.")
            with torch.no_grad():
                probe = torch.zeros(1, in_c, in_h, in_w)
                for i, conv_name in enumerate(self._conv_names):
                    bn_name = f"bn{i + 1}"
                    act_name = self._act_names[i]
                    pool_name = f"pool{i + 1}"
                    probe = getattr(self, conv_name)(probe)
                    probe = getattr(self, bn_name)(probe)
                    probe = getattr(self, act_name)(probe)
                    probe = getattr(self, pool_name)(probe)
                head_in_features = int(probe.flatten(1).size(1))
        self.head = nn.Linear(head_in_features, int(num_classes))

    @property
    def convs(self) -> list[nn.Conv2d]:
        return [getattr(self, name) for name in self._conv_names]

    @property
    def fc(self) -> nn.Linear:
        return self.head

    def forward(self, x: torch.Tensor):
        for i, conv_name in enumerate(self._conv_names):
            bn_name = f"bn{i + 1}"
            act_name = self._act_names[i]
            pool_name = f"pool{i + 1}"
            x = getattr(self, conv_name)(x)
            x = getattr(self, bn_name)(x)
            x = getattr(self, act_name)(x)
            x = getattr(self, pool_name)(x)
        x = self.gap(x).flatten(1) if self.gap is not None else x.flatten(1)
        logits = self.head(x)
        return self.output_activation(logits)


def _as_int_list(x: Sequence[int] | Iterable[int]) -> list[int]:
    return [int(v) for v in x]


def _broadcast(value, n: int) -> list:
    return [value for _ in range(n)]


class ConvNetClassifier(ClassicCNN):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: Sequence[int] | None = None,
        depth: int | None = 3,
        base_width: int = 32,
        width_factor: int = 2,
        kernel_sizes: int | Sequence[int] = 3,
        use_bn: bool = False,
        pool_every: int = 1,
        pools: Sequence[bool] | None = None,
        pool_kernel: int = 2,
        activation: str = "relu",
        output_activation: str = "identity",
        head_mode: str = "gap",
        input_shape: Sequence[int] | None = None,
    ):
        if channels is None:
            if depth is None:
                raise ValueError("If `channels` is None, you must provide `depth`.")
            d = int(depth)
            if d <= 0:
                raise ValueError("`depth` must be >= 1.")
            bw = int(base_width)
            wf = int(width_factor)
            if bw <= 0 or wf <= 0:
                raise ValueError("`base_width` and `width_factor` must be >= 1.")
            ch_list = [bw * (wf**i) for i in range(d)]
        else:
            ch_list = _as_int_list(channels)
            if len(ch_list) == 0:
                raise ValueError("`channels` must have at least one element.")
            d = len(ch_list)

        if isinstance(kernel_sizes, int):
            ks_list = _broadcast(int(kernel_sizes), d)
        else:
            ks_list = _as_int_list(kernel_sizes)
            if len(ks_list) != d:
                raise ValueError(f"`kernel_sizes` must have length {d}, got {len(ks_list)}.")

        if pools is not None:
            pool_list = [bool(p) for p in pools]
            if len(pool_list) != d:
                raise ValueError(f"`pools` must have length {d}, got {len(pool_list)}.")
        else:
            pe = int(pool_every)
            if pe <= 0:
                raise ValueError("`pool_every` must be >= 1.")
            pool_list = [(i % pe == pe - 1) for i in range(d)]

        super().__init__(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            widths=ch_list,
            kernel_sizes=ks_list,
            pools=pool_list,
            use_bn=bool(use_bn),
            pool_kernel=int(pool_kernel),
            activation=str(activation),
            output_activation=str(output_activation),
            head_mode=str(head_mode),
            input_shape=input_shape,
        )
