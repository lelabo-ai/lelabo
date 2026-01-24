# lab/models/resnet.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from .blocks import BlockModel, BlockSpec
except Exception:  # pragma: no cover
    from blocks import BlockModel, BlockSpec


class ResNetForCIFAR(BlockModel):
    """
    Torchvision ResNet adapté CIFAR + exposition blocks compatible SoftHebb.

    Blocks:
      - tous les nn.Conv2d du backbone (dans l'ordre d'itération de named_modules)
      - + "fc" (Linear) marqué is_output=True

    forward(return_cache=True):
      - retourne (logits, cache)
      - cache["block_inputs"][block_name] = input tensor au module (capturé via hook)
    """

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

        # Build blocks list once (Conv2d + fc)
        self._blocks: List[BlockSpec] = []
        self._block_modules: List[Tuple[str, nn.Module]] = []
        self._build_blocks()

    def _build_blocks(self) -> None:
        blocks: List[BlockSpec] = []
        mods: List[Tuple[str, nn.Module]] = []

        for name, m in self.resnet.named_modules():
            # Skip container root (empty name)
            if name == "":
                continue
            if isinstance(m, nn.Conv2d):
                blocks.append(BlockSpec(name=name, module=m, rep="gap", is_output=False))
                mods.append((name, m))

        # fc head (output)
        blocks.append(BlockSpec(name="fc", module=self.resnet.fc, rep="identity", is_output=True))
        mods.append(("fc", self.resnet.fc))

        self._blocks = blocks
        self._block_modules = mods

    def get_blocks(self) -> list[BlockSpec]:
        return list(self._blocks)

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        if not return_cache:
            return self.resnet(x)

        cache: Dict[str, Any] = {"block_inputs": {}}
        hooks = []

        def hook_fn(name: str):
            def _fn(module, inputs, output):
                if inputs and torch.is_tensor(inputs[0]):
                    cache["block_inputs"][name] = inputs[0].detach()
            return _fn

        # Register hooks only on blocks we expose
        for name, m in self._block_modules:
            hooks.append(m.register_forward_hook(hook_fn(name)))

        logits = self.resnet(x)

        for h in hooks:
            h.remove()

        return logits, cache


def build_resnet(num_classes: int = 100, resnet_type: str = "resnet34", pretrained: bool = False) -> ResNetForCIFAR:
    return ResNetForCIFAR(num_classes=num_classes, resnet_type=resnet_type, pretrained=pretrained)
