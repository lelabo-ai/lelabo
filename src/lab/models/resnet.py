# lab/models/resnet.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from .blocks import BlockModel, BlockSpec
except Exception:  # pragma: no cover
    from blocks import BlockModel, BlockSpec

from .registry import register_model


from .registry import register_model, ModelContext

@register_model("resnet18")
def build_resnet18(ctx: ModelContext, args):
    return ResNet(num_classes=ctx.num_classes, resnet_type="resnet18")

@register_model("resnet34")
def build_resnet34(ctx: ModelContext, args):
    return ResNet(num_classes=ctx.num_classes, resnet_type="resnet34")

@register_model("resnet50")
def build_resnet50(ctx: ModelContext, args):
    return ResNet(num_classes=ctx.num_classes, resnet_type="resnet50")


class ResNet(BlockModel):
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
            raise ValueError(f"Type de ResNet non supporté : {resnet_type}")

        # head
        num_ftrs = base.fc.in_features
        base.fc = nn.Linear(num_ftrs, self.num_classes)
        self.resnet = base

        self._blocks: List[BlockSpec] = []
        self._block_modules: List[Tuple[str, nn.Module]] = []
        self._build_blocks()

    def _build_blocks(self) -> None:
        blocks: List[BlockSpec] = []
        mods: List[Tuple[str, nn.Module]] = []

        # Expose residual blocks (BasicBlock / Bottleneck)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.resnet, layer_name)
            for i, blk in enumerate(layer):
                name = f"{layer_name}.{i}"
                blocks.append(BlockSpec(name=name, module=blk, rep="identity", is_output=False))
                mods.append((name, blk))

        # Head (output)
        blocks.append(BlockSpec(name="head", module=self.resnet.fc, rep="identity", is_output=True))
        mods.append(("head", self.resnet.fc))

        self._blocks = blocks
        self._block_modules = mods

    def get_blocks(self) -> list[BlockSpec]:
        return list(self._blocks)

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        if not return_cache:
            return self.resnet(x)

        cache: Dict[str, Any] = {"block_inputs": {}, "block_outputs": {}}
        hooks = []

        def hook_fn(name: str):
            def _fn(module, inputs, output):
                if inputs and torch.is_tensor(inputs[0]):
                    cache["block_inputs"][name] = inputs[0].detach()
                if torch.is_tensor(output):
                    cache["block_outputs"][name] = output.detach()
            return _fn

        for name, m in self._block_modules:
            hooks.append(m.register_forward_hook(hook_fn(name)))

        logits = self.resnet(x)

        for h in hooks:
            h.remove()

        return logits, cache
