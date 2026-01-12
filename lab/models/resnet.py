# lab/models/resnet.py
from __future__ import annotations

import torch
import torch.nn as nn


def build_resnet(
    name: str,
    num_classes: int,
    in_channels: int = 3,
    cifar_stem: bool = True,
    weights=None,
    expose_local_blocks: bool = True,
):
    import torchvision.models as tvm

    if not hasattr(tvm, name):
        raise ValueError(f"Unknown resnet variant: {name}")

    model_fn = getattr(tvm, name)
    base = model_fn(weights=weights)

    # CIFAR-friendly stem
    if cifar_stem:
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
    else:
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    base.fc = nn.Linear(base.fc.in_features, num_classes)

    return ResNetWithLocalBlocks(base) if expose_local_blocks else base


class ResNetWithLocalBlocks(nn.Module):
    """
    Wrapper autour torchvision ResNet.
    Expose:
      - local_blocks pour LocalProbeBlocks
      - forward(x, return_cache=True) -> (logits, {"block_inputs": {...}})
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

        self.local_blocks = [
            {"name": "stem",   "module": self._stem,        "rep": "gap", "is_output": False},
            {"name": "layer1", "module": self.base.layer1,  "rep": "gap", "is_output": False},
            {"name": "layer2", "module": self.base.layer2,  "rep": "gap", "is_output": False},
            {"name": "layer3", "module": self.base.layer3,  "rep": "gap", "is_output": False},
            {"name": "layer4", "module": self.base.layer4,  "rep": "gap", "is_output": False},
            {"name": "fc",     "module": self.base.fc,      "rep": "identity", "is_output": True},
        ]

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        x = s
