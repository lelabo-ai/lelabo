# lab/models/convnet.py
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .blocks import BlockModel, BlockSpec
except Exception:  # pragma: no cover
    from blocks import BlockModel, BlockSpec


class ConvBlock(nn.Module):
    """Conv -> (BN) -> ReLU -> (Pool)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        pool: bool = False,
        pool_kernel: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_kernel) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class ConvNetClassifier(BlockModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: tuple[int, int, int] = (32, 64, 128),
        use_bn: bool = True,
    ):
        super().__init__()
        c1, c2, c3 = [int(c) for c in channels]
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.use_bn = bool(use_bn)

        self.block1 = ConvBlock(self.in_channels, c1, use_bn=self.use_bn, pool=True)
        self.block2 = ConvBlock(c1, c2, use_bn=self.use_bn, pool=True)
        self.block3 = ConvBlock(c2, c3, use_bn=self.use_bn, pool=False)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, self.num_classes)

    def get_blocks(self) -> list[BlockSpec]:
        """
        IMPORTANT:
        - On expose les vrais modules apprenables (Conv2d / Linear).
        - On n'expose plus ConvBlock (wrapper), sinon beaucoup de règles ne les entraînent pas.
        """
        return [
            BlockSpec(name="conv1", module=self.block1.conv, rep="identity", is_output=False),
            BlockSpec(name="conv2", module=self.block2.conv, rep="identity", is_output=False),
            BlockSpec(name="conv3", module=self.block3.conv, rep="identity", is_output=False),
            BlockSpec(name="head", module=self.fc, rep="identity", is_output=True),
        ]

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        cache = {"block_inputs": {}} if return_cache else None

        # ---- conv1 input (avant conv1)
        if return_cache:
            cache["block_inputs"]["conv1"] = x
        x = self.block1(x)

        # ---- conv2 input
        if return_cache:
            cache["block_inputs"]["conv2"] = x
        x = self.block2(x)

        # ---- conv3 input
        if return_cache:
            cache["block_inputs"]["conv3"] = x
        x = self.block3(x)

        # ---- head input
        x = self.pool(x).flatten(1)
        if return_cache:
            cache["block_inputs"]["head"] = x
        logits = self.fc(x)

        return (logits, cache) if return_cache else logits
