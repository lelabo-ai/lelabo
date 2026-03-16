"""Tensor and vision transform helpers for dataset pipelines."""

from __future__ import annotations

import torch


def apply_relative_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma * torch.abs(x)


def apply_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


class AddRelativeNoise:
    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return apply_relative_noise(x, self.sigma)


class AddGaussianNoise:
    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return apply_gaussian_noise(x, self.sigma)


class Flatten:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)
"""Tensor and vision transform helpers for dataset pipelines."""
