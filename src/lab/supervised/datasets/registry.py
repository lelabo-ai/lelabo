from __future__ import annotations

from pathlib import Path

from lab.core.registry import Registry
from ...core.utils.capsule_plugins import load_capsule_plugins, load_installed_capsule_plugins

DATASET_REGISTRY = Registry("datasets", package="lab.supervised.datasets")


def register_dataset(name: str):
    return DATASET_REGISTRY.register(name)


def _load_dataset_capsule_plugins(*, capsules_dir: Path | None = None) -> None:
    load_capsule_plugins(kinds=("datasets",))
    load_installed_capsule_plugins(kinds=("datasets",), capsules_dir=capsules_dir)


def get_dataset(name: str, **kwargs):
    if not DATASET_REGISTRY.names():
        DATASET_REGISTRY.discover()
    _load_dataset_capsule_plugins()
    builder = DATASET_REGISTRY.get(name)
    return builder(**kwargs)


def get_dataset_names(*, capsules_dir: Path | None = None):
    if not DATASET_REGISTRY.names():
        DATASET_REGISTRY.discover()
    _load_dataset_capsule_plugins(capsules_dir=capsules_dir)
    return DATASET_REGISTRY.names()
