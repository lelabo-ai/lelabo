from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from lab.core.registry import Registry  # si tu utilises le Registry générique

MODEL_REGISTRY = Registry("models", package="lab.models.imported")

@dataclass
class ModelContext:
    dataset: str
    num_classes: int
    in_dim: int | None = None          # pour MLP
    in_channels: int | None = None     # pour CNN/ResNet
    input_shape: tuple[int, ...] | None = None
    extra: dict[str, Any] | None = None

def register_model(name: str):
    return MODEL_REGISTRY.register(name)

def build_model(name: str, ctx: ModelContext, args):
    if not MODEL_REGISTRY.names():
        MODEL_REGISTRY.discover()
    builder = MODEL_REGISTRY.get(name)
    return builder(ctx, args)

def get_model_names():
    if not MODEL_REGISTRY.names():
        MODEL_REGISTRY.discover()
    return MODEL_REGISTRY.names()
