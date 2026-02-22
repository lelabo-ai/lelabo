# lab/models/resnet.py
from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks import LeModule, BlockSpec
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


class ResNet(LeModule):
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

    def get_blocks(self) -> list[BlockSpec]:
        specs: list[BlockSpec] = []
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(self.resnet, layer_name)
            for i, block in enumerate(layer):
                specs.append(
                    BlockSpec(
                        name=f"{layer_name}.{i}",
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
        return self.resnet(x)
