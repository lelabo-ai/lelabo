from __future__ import annotations

from lab.core.registry import Registry

DATASET_REGISTRY = Registry("datasets", package="lab.supervised.datasets")


def register_dataset(name: str):
    return DATASET_REGISTRY.register(name)


def get_dataset(name: str, **kwargs):
    if not DATASET_REGISTRY.names():
        DATASET_REGISTRY.discover()
    builder = DATASET_REGISTRY.get(name)
    return builder(**kwargs)


def get_dataset_names():
    if not DATASET_REGISTRY.names():
        DATASET_REGISTRY.discover()
    return DATASET_REGISTRY.names()
