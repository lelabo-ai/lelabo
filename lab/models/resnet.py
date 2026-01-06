# models/resnet.py
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

    if not expose_local_blocks:
        return base

    return ResNetWithLocalBlocks(base)


class ResNetWithLocalBlocks(nn.Module):
    """
    Thin wrapper around torchvision ResNet that exposes:
      - local_blocks: list of block specs for LocalProbeBlocks
      - forward(x, return_cache=True): returns (logits, cache) with block_inputs
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

        # expose blocks for your LocalProbeBlocks algo
        self.local_blocks = [
            {"name": "stem",   "module": self._stem,  "rep": "gap",      "is_output": False},
            {"name": "layer1", "module": self.base.layer1, "rep": "gap", "is_output": False},
            {"name": "layer2", "module": self.base.layer2, "rep": "gap", "is_output": False},
            {"name": "layer3", "module": self.base.layer3, "rep": "gap", "is_output": False},
            {"name": "layer4", "module": self.base.layer4, "rep": "gap", "is_output": False},
            # For the head, we treat it as output; module takes flat features -> logits
            {"name": "fc",     "module": self.base.fc, "rep": "identity", "is_output": True},
        ]

    def _stem(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        return x

    def forward(self, x, return_cache: bool = False):
        cache = {"block_inputs": {}} if return_cache else None

        if return_cache:
            cache["block_inputs"]["stem"] = x
        x = self._stem(x)

        if return_cache:
            cache["block_inputs"]["layer1"] = x
        x = self.base.layer1(x)

        if return_cache:
            cache["block_inputs"]["layer2"] = x
        x = self.base.layer2(x)

        if return_cache:
            cache["block_inputs"]["layer3"] = x
        x = self.base.layer3(x)

        if return_cache:
            cache["block_inputs"]["layer4"] = x
        x = self.base.layer4(x)

        # global average pool + flatten
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)

        if return_cache:
            cache["block_inputs"]["fc"] = x
        logits = self.base.fc(x)

        return (logits, cache) if return_cache else logits
