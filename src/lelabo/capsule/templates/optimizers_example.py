"""
example_optimizer.py

Minimal example showing:

1) How to implement a custom optimizer
2) How to inherit from torch.optim.Optimizer
3) How to register it
4) What step() is expected to do

This example implements a SGD optimizer.

This is strictly the same optimizer as in torch.
"""

from __future__ import annotations
from typing import Iterable

import torch

from lab.optim.registry import register_optimizer

class ExampleSGD(torch.optim.Optimizer):
    """
    Minimal SGD optimizer.

    Update rule:
        p = p - lr * grad
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive.")

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        closure (optional):
            A function that reevaluates the model and returns the loss.
            Used for some optimizers (e.g., LBFGS).
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                p.data -= lr * grad

        return loss

# Register the optimizer so it can be used from the CLI.
# You need to uncomment the following line to register it.
#@register_optimizer("custom_sgd")
def build_example_sgd(model, args):
    """
    Factory used by:

        lelabo train --optimizer example_sgd

    Receives:
        - model
        - CLI args
    """

    lr = getattr(args, "lr", 1e-3)
    return ExampleSGD(
        model.parameters(),
        lr=lr,
    )