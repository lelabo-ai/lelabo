"""
example.py

Purpose
-------
Add a dataset builder to this capsule.

Contract
--------
1. Register a builder with `@register_dataset("example_mnist")`
2. The builder returns a `DataBundle`
3. The bundle defines train/val/test loaders plus dataset metadata

Where params come from
----------------------
For the full add-a-dataset procedure, read `resources/EXTENSION_RECIPES.md`.
Dataset builders receive keyword args directly.
For the exact `DataBundle` fields, read `resources/LELABO_REFERENCE.md`.
For the param mapping, read `resources/PARAM_FLOW.md`.

Official example
----------------
`example_mnist` downloads MNIST, creates train/val/test loaders, and returns the
metadata needed by models.

Config snippet
--------------
[dataset]
name = "example_mnist"

[dataset.params]
batch_size = 128
flatten = true

How to activate
---------------
Uncomment `@register_dataset("example_mnist")`.

How to test
-----------
lelabo registries datasets
lelabo train supervised --dataset example_mnist --model mlp

Common errors
-------------
- The builder must return a `DataBundle`.
- If torchvision is not installed, this example raises a clear runtime error.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import random_split

from lelabo.supervised.datasets.base import DataBundle, dataset_to_tensors, make_loader
from lelabo.supervised.datasets.paths import dataset_dir
from lelabo.supervised.datasets.registry import register_dataset

try:  # optional dependency for this example only
    from torchvision import transforms
    from torchvision.datasets import MNIST
except Exception:  # pragma: no cover - handled at runtime in factory
    MNIST = None
    transforms = None


# @register_dataset("example_mnist")
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
    if MNIST is None or transforms is None:
        raise RuntimeError(
            "example_mnist requires torchvision. Install it to use this dataset template."
        )

    torch.manual_seed(seed)

    tfms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    if flatten:
        tfms.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(tfms)

    data_root = str(dataset_dir("mnist"))
    train_full = MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = MNIST(root=data_root, train=False, download=True, transform=transform)

    n_total = len(train_full)
    n_val = int(val_frac * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(train_full, [n_train, n_val])

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
