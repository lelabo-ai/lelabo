# lab/models/convnet.py
from __future__ import annotations

import torch
import torch.nn as nn


class ConvNetClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        self.local_blocks = [
            {"name": "block1", "module": self.block1, "rep": "gap", "is_output": False},
            {"name": "block2", "module": self.block2, "rep": "gap", "is_output": False},
            {"name": "block3", "module": self.block3, "rep": "gap", "is_output": False},
            {"name": "fc",     "module": self.fc,     "rep": "identity", "is_output": True},
        ]

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        cache = {"block_inputs": {}} if return_cache else None

        if return_cache: cache["block_inputs"]["block1"] = x
        x = self.block1(x)

        if return_cache: cache["block_inputs"]["block2"] = x
        x = self.block2(x)

        if return_cache: cache["block_inputs"]["block3"] = x
        x = self.block3(x)

        x = self.pool(x).flatten(1)

        if return_cache: cache["block_inputs"]["fc"] = x
        logits = self.fc(x)

        return (logits, cache) if return_cache else logits
