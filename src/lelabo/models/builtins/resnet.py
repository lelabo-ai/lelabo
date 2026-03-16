# lab/models/resnet.py
"""Residual network baselines for vision classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks import BlockSpec
from ..registry import ModelContext, register_model


@register_model("resnet18")
def build_resnet18(ctx: ModelContext, args):
    return ResNet(num_classes=ctx.num_classes, resnet_type="resnet18")


@register_model("resnet34")
def build_resnet34(ctx: ModelContext, args):
    return ResNet(num_classes=ctx.num_classes, resnet_type="resnet34")


@register_model("resnet50")
def build_resnet50(ctx: ModelContext, args):
    return ResNet(num_classes=ctx.num_classes, resnet_type="resnet50")


class ResNet(nn.Module):
    class _StemWithFirstResidual(nn.Module):
        def __init__(
            self,
            *,
            conv1: nn.Module,
            bn1: nn.Module,
            relu: nn.Module,
            maxpool: nn.Module,
            first_block: nn.Module,
        ) -> None:
            super().__init__()
            self.conv1 = conv1
            self.bn1 = bn1
            self.relu = relu
            self.maxpool = maxpool
            self.first_block = first_block

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.first_block(x)
            return x

    def __init__(self, num_classes: int = 100, resnet_type: str = "resnet34", pretrained: bool = True):
        super().__init__()

        import torchvision.models as models

        resnet_type = str(resnet_type).lower()
        self.resnet_type = resnet_type
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)

        if resnet_type == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
        elif resnet_type == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet34(weights=weights)
        elif resnet_type == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        num_ftrs = base.fc.in_features
        base.fc = nn.Linear(num_ftrs, self.num_classes)
        self.resnet = base

        # Runtime block decomposition used by local rules.
        self._runtime_block_names: list[str] = []
        self._runtime_blocks = nn.ModuleList()

        has_stem = all(hasattr(self.resnet, name) for name in ("conv1", "bn1", "relu", "maxpool"))
        has_layer1 = hasattr(self.resnet, "layer1") and len(self.resnet.layer1) > 0
        self._stem_in_first_block = bool(has_stem and has_layer1)

        if self._stem_in_first_block:
            self._runtime_block_names.append("layer1.0")
            self._runtime_blocks.append(
                ResNet._StemWithFirstResidual(
                    conv1=self.resnet.conv1,
                    bn1=self.resnet.bn1,
                    relu=self.resnet.relu,
                    maxpool=self.resnet.maxpool,
                    first_block=self.resnet.layer1[0],
                )
            )

        for stage_name in ("layer1", "layer2", "layer3", "layer4"):
            stage = getattr(self.resnet, stage_name)
            start_idx = 1 if (self._stem_in_first_block and stage_name == "layer1") else 0
            for i in range(start_idx, len(stage)):
                self._runtime_block_names.append(f"{stage_name}.{i}")
                self._runtime_blocks.append(stage[i])

    def declare_blocks(self) -> list[BlockSpec]:
        specs: list[BlockSpec] = []
        for name, block in zip(self._runtime_block_names, self._runtime_blocks):
            specs.append(
                BlockSpec(
                    name=name,
                    module=block,
                    rep="gap",
                    is_output=False,
                )
            )
        specs.append(
            BlockSpec(
                name="head",
                module=self.resnet.fc,
                rep="identity",
                is_output=True,
            )
        )
        return specs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self._runtime_blocks:
            x = block(x)
        pool = getattr(self.resnet, "avgpool", None)
        if pool is None:
            pool = getattr(self.resnet, "gap", None)
        if isinstance(pool, nn.Module):
            x = pool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x
