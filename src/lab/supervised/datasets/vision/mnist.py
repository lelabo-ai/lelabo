from __future__ import annotations

from typing import Optional
import torch
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision import transforms

from ..base import DataBundle, dataset_to_tensors, make_loader
from ..paths import dataset_dir
from ..registry import register_dataset
from ..splits import split_train_val
from ..transforms import AddRelativeNoise, Flatten


def _build_transform(*, add_noise: bool, flatten: bool, noise_sigma: float) -> transforms.Compose:
    tfms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    if add_noise and noise_sigma > 0.0:
        tfms.append(AddRelativeNoise(noise_sigma))
    if flatten:
        tfms.append(Flatten())
    return transforms.Compose(tfms)


def _maybe_subset(ds, max_items: Optional[int], seed: int) -> object:
    """Return ds or a deterministic Subset(ds, idx) of size max_items."""
    if max_items is None:
        return ds
    if max_items <= 0:
        raise ValueError(f"max_items must be None or a positive int, got {max_items}")
    n = len(ds)
    if max_items >= n:
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:max_items].tolist()
    return Subset(ds, idx)


@register_dataset("mnist")
def make_mnist_dataset(
    *,
    batch_size: int = 128,
    seed: int = 42,
    val_frac: float = 0.0,
    flatten: bool = True,
    input_noise_dataset: float = 0.0,
    noise_on_test: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    # new: cap sizes (None => keep full / "max")
    train_max: Optional[int] = 1000,
    val_max: Optional[int] = 200,
    test_max: Optional[int] = 1000,
    **_: object,
) -> DataBundle:
    data_root = str(dataset_dir("mnist"))

    train_tfm = _build_transform(add_noise=(input_noise_dataset > 0.0), flatten=flatten, noise_sigma=input_noise_dataset)
    val_tfm = _build_transform(add_noise=(input_noise_dataset > 0.0), flatten=flatten, noise_sigma=input_noise_dataset)
    test_tfm = _build_transform(add_noise=(input_noise_dataset > 0.0 and noise_on_test), flatten=flatten, noise_sigma=input_noise_dataset)

    train_full = MNIST(root=data_root, train=True, download=True, transform=train_tfm)
    val_full = MNIST(root=data_root, train=True, download=False, transform=val_tfm)
    test_full = MNIST(root=data_root, train=False, download=True, transform=test_tfm)

    n = len(train_full)
    tr_idx, va_idx = split_train_val(n, val_frac, seed)

    train_ds = Subset(train_full, tr_idx.tolist())
    val_ds = Subset(val_full, va_idx.tolist()) if va_idx.numel() > 0 else None

    # apply caps (use different seeds so subsets differ across splits reproducibly)
    train_ds = _maybe_subset(train_ds, train_max, seed=seed + 1)
    if val_ds is not None:
        val_ds = _maybe_subset(val_ds, val_max, seed=seed + 2)
    test_ds = _maybe_subset(test_full, test_max, seed=seed + 3)

    train_loader = make_loader(
        train_ds, batch_size=batch_size, shuffle=True, seed=seed, seed_scope="mnist.train",
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = (
        make_loader(val_ds, batch_size=batch_size, shuffle=False, seed=seed, seed_scope="mnist.val",
                    num_workers=num_workers, pin_memory=pin_memory)
        if val_ds is not None else None
    )
    test_loader = make_loader(
        test_ds, batch_size=batch_size, shuffle=False, seed=seed, seed_scope="mnist.test",
        num_workers=num_workers, pin_memory=pin_memory
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
