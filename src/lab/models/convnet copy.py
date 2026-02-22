# lab/models/convnet.py
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

try:
    from .blocks import LeModule, BlockSpec
except Exception:  # pragma: no cover
    from blocks import LeModule, BlockSpec

from .registry import register_model, ModelContext

@register_model("cnn")
def build_cnn(ctx: ModelContext, args):
    if ctx.in_channels is None:
        raise ValueError("CNN needs ctx.in_channels")
    return ConvNetClassifier(in_channels=ctx.in_channels, num_classes=ctx.num_classes)


class ConvBlock(nn.Module):
    """Conv -> (BN) -> ReLU -> (Pool). Cache is taken right after Conv2d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        use_bn: bool = True,
        pool: bool = False,
        pool_kernel: int = 2,
    ):
        super().__init__()
        if padding is None:
            # "same-ish" padding for odd kernels when stride=1
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_kernel) if pool else nn.Identity()

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        cache = None
        x = self.conv(x)
        if return_cache:
            cache = x.detach().clone()
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return (x, cache) if return_cache else x



def _as_int_list(x: Sequence[int] | Iterable[int]) -> list[int]:
    return [int(v) for v in x]


def _broadcast(value, n: int) -> list:
    return [value for _ in range(n)]


class ConvNetClassifier(LeModule):
    """
    Modular ConvNet.

    You have two ways to choose the widths:
    - Option A: provide `channels=(c1, c2, ..., cL)`
    - Option B: set `channels=None` and provide `depth`, `base_width`, `width_factor`
      which generates: [base_width, base_width*width_factor, base_width*width_factor^2, ...]

    Pooling can be configured with:
    - `pools=(True/False per layer)` OR
    - `pool_every=k` meaning pool every k-th layer (default k=1 => pool after every layer)
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 10,
        # Option A: explicit widths
        channels: Sequence[int] | None = None,
        # Option B: generated widths (paper-like)
        depth: int | None = 3,
        base_width: int = 96,
        width_factor: int = 4,
        # Convolution / pooling config
        kernel_sizes: int | Sequence[int] = 3,
        use_bn: bool = False,
        pool_every: int = 1,  # pool every k layers (1 = every layer)
        pools: Sequence[bool] | None = None,  # overrides pool_every if provided
        pool_kernel: int = 2,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.use_bn = bool(use_bn)

        # ---- decide channels
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

        # ---- kernel sizes (broadcast or per-layer)
        if isinstance(kernel_sizes, int):
            ks_list = _broadcast(int(kernel_sizes), d)
        else:
            ks_list = _as_int_list(kernel_sizes)
            if len(ks_list) != d:
                raise ValueError(f"`kernel_sizes` must have length {d}, got {len(ks_list)}.")

        # ---- pooling pattern
        if pools is not None:
            pool_list = [bool(p) for p in pools]
            if len(pool_list) != d:
                raise ValueError(f"`pools` must have length {d}, got {len(pool_list)}.")
        else:
            pe = int(pool_every)
            if pe <= 0:
                raise ValueError("`pool_every` must be >= 1.")
            # pool on layers: pe-1, 2*pe-1, 3*pe-1, ...
            pool_list = [(i % pe == pe - 1) for i in range(d)]

        # IMPORTANT: do NOT name this attribute `blocks` (LeModule has a property called blocks).
        self.conv_blocks = nn.ModuleList()
        prev = self.in_channels
        for c_out, k, do_pool in zip(ch_list, ks_list, pool_list):
            self.conv_blocks.append(
                ConvBlock(
                    prev,
                    int(c_out),
                    kernel_size=int(k),
                    use_bn=self.use_bn,
                    pool=bool(do_pool),
                    pool_kernel=int(pool_kernel),
                )
            )
            prev = int(c_out)

        # ---- head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(prev, self.num_classes)

        # for naming / caching
        self._conv_names = [f"conv{i}" for i in range(1, len(self.conv_blocks) + 1)]

    def get_blocks(self) -> list[BlockSpec]:
        specs: list[BlockSpec] = []
        for name, block in zip(self._conv_names, self.conv_blocks):
            specs.append(BlockSpec(name=name, module=block, rep="identity", is_output=False))
        specs.append(BlockSpec(name="head", module=self.fc, rep="identity", is_output=True))
        return specs

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        cache = {"block_inputs": {}, "block_outputs": {}} if return_cache else None

        for name, block in zip(self._conv_names, self.conv_blocks):
            if return_cache:
                cache["block_inputs"][name] = x
            if return_cache:
                x, cache_block = block(x, return_cache=return_cache)
            else:
                x = block(x, return_cache=return_cache)
            if return_cache:
                cache["block_outputs"][name] = cache_block

        x = self.avgpool(x).flatten(1)
        if return_cache:
            cache["block_inputs"]["head"] = x
        logits = self.fc(x)
        if return_cache:
            cache["block_outputs"]["head"] = logits

        return (logits, cache) if return_cache else logits
