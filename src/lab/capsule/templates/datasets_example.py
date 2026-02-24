"""
example_dataset.py

Minimal example showing:

1) How to register a dataset
2) How to return a DataBundle
3) How train/val/test loaders are built
4) How in_dim / input_shape are defined

This example uses MNIST.
"""

from __future__ import annotations
from typing import Optional

import torch
from torch.utils.data import random_split

from lab.supervised.datasets.base import DataBundle, dataset_to_tensors, make_loader
from lab.supervised.datasets.registry import register_dataset
from lab.supervised.datasets.paths import dataset_dir

try:  # optional dependency for this example only
    from torchvision.datasets import MNIST
    from torchvision import transforms
except Exception:  # pragma: no cover - handled at runtime in factory
    MNIST = None
    transforms = None


# Dataset registration
# To make this dataset available in the command line interface, we need to register it.
# Please uncomment the @register_dataset decorator if you want to use this dataset
# in the command line interface.

#@register_dataset("example_mnist")
def make_example_mnist(
    *,
    batch_size: int = 128,
    seed: int = 42,
    val_frac: float = 0.1,
    flatten: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    **_: object,
) -> DataBundle:
    """
    Factory used by:

        lelabo train supervised --dataset example_mnist

    Must return a DataBundle.
    """
    if MNIST is None or transforms is None:
        raise RuntimeError(
            "example_mnist requires torchvision. Install it to use this dataset template."
        )

    torch.manual_seed(seed)

    # Transforms

    tfms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    if flatten:
        tfms.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(tfms)

    # Load datasets

    data_root = str(dataset_dir("mnist"))

    train_full = MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = MNIST(root=data_root, train=False, download=True, transform=transform)

    # Train / validation split

    n_total = len(train_full)
    n_val = int(val_frac * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(train_full, [n_train, n_val])

    # DataLoaders

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        seed_scope="example_mnist.train",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope="example_mnist.val",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope="example_mnist.test",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Metadata for models

    x_test, y_test = dataset_to_tensors(test_ds)

    if flatten:
        in_dim: Optional[int] = int(x_test.shape[1])
        input_shape = None
    else:
        in_dim = None
        input_shape = (1, 28, 28)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=10,
        in_dim=in_dim,
        input_shape=input_shape,
        x_test=x_test,
        y_test=y_test,
    )
