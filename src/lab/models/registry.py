from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.core.registry import Registry  # si tu utilises le Registry générique
from ..core.utils.capsule_plugins import load_capsule_plugins, load_installed_capsule_plugins

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

def _load_model_capsule_plugins(*, capsules_dir: Path | None = None) -> None:
    load_capsule_plugins(kinds=("models",))
    load_installed_capsule_plugins(kinds=("models",), capsules_dir=capsules_dir)

def build_model(name: str, ctx: ModelContext, args):
    if not MODEL_REGISTRY.names():
        MODEL_REGISTRY.discover()
    _load_model_capsule_plugins()
    builder = MODEL_REGISTRY.get(name)
    return builder(ctx, args)

def get_model_names(*, capsules_dir: Path | None = None):
    if not MODEL_REGISTRY.names():
        MODEL_REGISTRY.discover()
    _load_model_capsule_plugins(capsules_dir=capsules_dir)
    return MODEL_REGISTRY.names()
