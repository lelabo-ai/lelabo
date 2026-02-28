# lab/models/convnet.py
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import BlockSpec
from ..registry import ModelContext, register_model


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
    )


class ClassicCNN(nn.Module):
    """CNN classique: (Conv -> BN -> ReLU -> Pool) x N -> GAP -> Linear."""

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
    ):
        super().__init__()
        self.activation_name = "relu"
        if not widths:
            raise ValueError("ClassicCNN requires at least one conv width.")
        if not (len(widths) == len(kernel_sizes) == len(pools)):
            raise ValueError("widths, kernel_sizes, pools must share the same length.")

        self._conv_names: list[str] = []
        c_in = int(in_channels)
        for i, (w, k, do_pool) in enumerate(zip(widths, kernel_sizes, pools), start=1):
            conv_name = f"conv{i}"
            bn_name = f"bn{i}"
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
            setattr(self, pool_name, nn.MaxPool2d(int(pool_kernel)) if bool(do_pool) else nn.Identity())

            self._conv_names.append(conv_name)
            c_in = int(w)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c_in, int(num_classes))

    @property
    def convs(self) -> list[nn.Conv2d]:
        return [getattr(self, name) for name in self._conv_names]

    @property
    def fc(self) -> nn.Linear:
        return self.head

    def get_blocks(self) -> list[BlockSpec]:
        specs: list[BlockSpec] = []
        for name in self._conv_names:
            specs.append(BlockSpec(name=name, module=getattr(self, name), rep="gap", is_output=False))
        specs.append(BlockSpec(name="head", module=self.head, rep="identity", is_output=True))
        return specs

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        if not return_cache:
            for i, conv_name in enumerate(self._conv_names, start=1):
                bn_name = f"bn{i}"
                pool_name = f"pool{i}"
                x = getattr(self, conv_name)(x)
                x = getattr(self, bn_name)(x)
                x = F.relu(x, inplace=True)
                x = getattr(self, pool_name)(x)
            x = self.gap(x).flatten(1)
            return self.head(x)

        # --- cache path
        block_inputs: dict[str, torch.Tensor] = {}
        block_preacts: dict[str, torch.Tensor] = {}
        block_outputs: dict[str, torch.Tensor] = {}
        steps: list[dict[str, object]] = []

        for i, conv_name in enumerate(self._conv_names, start=1):
            name = conv_name
            bn_name = f"bn{i}"
            pool_name = f"pool{i}"

            x_in = x
            block_inputs[name] = x_in.detach()

            u = getattr(self, conv_name)(x_in)          # conv output
            u_bn = getattr(self, bn_name)(u)            # this is what ReLU sees => "preact"
            h = F.relu(u_bn, inplace=False)             # avoid inplace messing with cached tensors
            x = getattr(self, pool_name)(h)

            block_preacts[name] = u_bn.detach()
            block_outputs[name] = h.detach()

            steps.append(
                {
                    "name": name,
                    "type": "Conv2d",
                    "is_output": False,
                    "x_in": block_inputs[name],
                    "preact": block_preacts[name],
                    "out": block_outputs[name],
                }
            )

        # head
        feats = self.gap(x).flatten(1)
        block_inputs["head"] = feats.detach()
        logits = self.head(feats)
        block_preacts["head"] = logits.detach()
        block_outputs["head"] = logits.detach()
        steps.append(
            {
                "name": "head",
                "type": "Linear",
                "is_output": True,
                "x_in": block_inputs["head"],
                "preact": block_preacts["head"],
                "out": block_outputs["head"],
            }
        )

        cache = {
            "cache_version": "standard.v1",
            "block_inputs": block_inputs,
            "block_outputs": block_outputs,   # post-activation for conv blocks here
            "block_preacts": block_preacts,
            "block_specs_runtime": self.get_blocks(),
            "steps": steps,
        }
        return logits, normalize_standard_cache(cache)


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
        )
