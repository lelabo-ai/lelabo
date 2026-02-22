# lab/models/deep_softhebb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blocks import LeModule, BlockSpec
except Exception:  # pragma: no cover
    from blocks import LeModule, BlockSpec

from .registry import register_model, ModelContext


class Triangle(nn.Module):
    def __init__(self, power: float = 1.0, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
        self.power = float(power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: matches your demo: mean over channel dim, using .data semantics effectively
        x = x - torch.mean(x.data, dim=1, keepdim=True)
        return F.relu(x, inplace=self.inplace) ** self.power


class SoftHebbConv2d(nn.Module):
    """
    Same spirit as your demo SoftHebbConv2d, but:
      - forward returns only weighted_input (u)
      - NO update happens here (updates are performed by the UpdateRule)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        t_invert: float = 12.0,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        assert groups == 1, "Simple implementation does not support groups > 1."
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.kernel_size = (int(kernel_size), int(kernel_size))
        self.stride = (int(stride), int(stride))
        self.dilation = (int(dilation), int(dilation))
        self.groups = int(groups)

        self.padding_mode = str(padding_mode)
        self.F_padding = (int(padding), int(padding), int(padding), int(padding))

        # demo init
        weight_range = 25.0 / ((in_channels / groups) * kernel_size * kernel_size) ** 0.5
        self.weight = nn.Parameter(weight_range * torch.randn(out_channels, in_channels // groups, *self.kernel_size))
        self.t_invert = float(t_invert)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.F_padding, self.padding_mode)
        u = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)
        return u

    def padded_input(self, x: torch.Tensor) -> torch.Tensor:
        # convenience for the update rule (exactly matches forward padding)
        return F.pad(x, self.F_padding, self.padding_mode)


class SoftHebbBlock(nn.Module):
    """
    Block = BN(affine=False) -> SoftHebbConv2d -> Triangle -> Pool
    Cache point: right after conv (u), like your framework expects.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        padding: int,
        t_invert: float,
        tri_power: float,
        pool: nn.Module,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch, affine=False)
        self.conv = SoftHebbConv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding,
            t_invert=t_invert,
        )
        self.act = Triangle(power=tri_power)
        self.pool = pool

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        cache_u = None
        x = self.bn(x)
        u = self.conv(x)
        if return_cache:
            cache_u = u.detach()
        x = self.act(u)
        x = self.pool(x)
        return (x, cache_u) if return_cache else x


class DeepSoftHebbClassifier(LeModule):
    """
    Mirrors your demo architecture:
      - bn1 (affine=False) + conv1(3->96,k5,p2,t=1) + Triangle(0.7) + MaxPool(k4,s2,p1)
      - bn2 + conv2(96->384,k3,p1,t=0.65) + Triangle(1.4) + MaxPool(k4,s2,p1)
      - bn3 + conv3(384->1536,k3,p1,t=0.25) + Triangle(1.0) + AvgPool(k2,s2)
      - Flatten -> Dropout(0.5) -> Linear(24576->num_classes)
    """

    def __init__(self, *, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        self.block1 = SoftHebbBlock(
            in_ch=self.in_channels,
            out_ch=96,
            kernel_size=5,
            padding=2,
            t_invert=1.0,
            tri_power=0.7,
            pool=nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
        )
        self.block2 = SoftHebbBlock(
            in_ch=96,
            out_ch=384,
            kernel_size=3,
            padding=1,
            t_invert=0.65,
            tri_power=1.4,
            pool=nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
        )
        self.block3 = SoftHebbBlock(
            in_ch=384,
            out_ch=1536,
            kernel_size=3,
            padding=1,
            t_invert=0.25,
            tri_power=1.0,
            pool=nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        # demo uses fixed 24576 for CIFAR10 with that pooling path
        self.fc = nn.Linear(24576, self.num_classes)

        # demo init for classifier weights
        with torch.no_grad():
            self.fc.weight.data = 0.11048543456039805 * torch.rand_like(self.fc.weight.data)

        self._block_names = ["conv1", "conv2", "conv3"]

    def get_blocks(self) -> List[BlockSpec]:
        # expose blocks to UpdateRule
        return [
            BlockSpec(name="conv1", module=self.block1, rep="identity", is_output=False),
            BlockSpec(name="conv2", module=self.block2, rep="identity", is_output=False),
            BlockSpec(name="conv3", module=self.block3, rep="identity", is_output=False),
            BlockSpec(name="head", module=self.fc, rep="identity", is_output=True),
        ]

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        cache: Optional[Dict[str, Dict[str, Any]]] = None
        if return_cache:
            cache = {"block_inputs": {}, "block_outputs": {}}

        # block 1
        if return_cache:
            cache["block_inputs"]["conv1"] = x
            x, u1 = self.block1(x, return_cache=return_cache)
            cache["block_outputs"]["conv1"] = u1
        else:
            x = self.block1(x, return_cache=return_cache)

        # block 2
        if return_cache:
            cache["block_inputs"]["conv2"] = x
            x, u2 = self.block2(x, return_cache=return_cache)
            cache["block_outputs"]["conv2"] = u2
        else:
            x = self.block2(x, return_cache=return_cache)

        # block 3
        if return_cache:
            cache["block_inputs"]["conv3"] = x
            x, u3 = self.block3(x, return_cache=return_cache)
            cache["block_outputs"]["conv3"] = u3
        else:
            x = self.block3(x, return_cache=return_cache)

        x = self.flatten(x)
        x = self.dropout(x)
        if return_cache:
            cache["block_inputs"]["head"] = x
        logits = self.fc(x)
        if return_cache:
            cache["block_outputs"]["head"] = logits
            return logits, cache
        return logits


@register_model("deephebb")
def build_deep_softhebb_demo(ctx: ModelContext, args):
    if ctx.in_channels is None:
        raise ValueError("deep_softhebb_demo needs ctx.in_channels")
    return DeepSoftHebbClassifier(in_channels=ctx.in_channels, num_classes=ctx.num_classes)
