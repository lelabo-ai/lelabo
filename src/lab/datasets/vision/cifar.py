from __future__ import annotations

from typing import Optional, Tuple
from torch.utils.data import Subset
from torchvision import datasets, transforms

from ..base import DataBundle, dataset_to_tensors, make_loader
from ..paths import dataset_dir
from ..registry import register_dataset
from ..splits import split_train_val
from ..transforms import AddRelativeNoise, Flatten


def _build_transform(
    *,
    augment: bool,
    add_noise: bool,
    flatten: bool,
    noise_sigma: float,
) -> transforms.Compose:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    tfms = []
    if augment:
        tfms += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    if add_noise and noise_sigma > 0.0:
        tfms.append(AddRelativeNoise(noise_sigma))

    if flatten:
        tfms.append(Flatten())

    return transforms.Compose(tfms)


def _make_cifar_dataset(
    *,
    dataset: str,
    batch_size: int = 128,
    seed: int = 42,
    val_frac: float = 0.0,
    flatten: bool = False,
    input_noise_dataset: float = 0.0,
    noise_on_test: bool = False,
    augment: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    **_: object,
) -> DataBundle:
    dataset = str(dataset).lower()
    if dataset not in {"cifar10", "cifar100"}:
        raise ValueError(f"Unknown CIFAR dataset: {dataset}")

    data_root = str(dataset_dir(dataset))

    train_tfm = _build_transform(
        augment=augment,
        add_noise=(input_noise_dataset > 0.0),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
    )
    val_tfm = _build_transform(
        augment=False,
        add_noise=(input_noise_dataset > 0.0),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
    )
    test_tfm = _build_transform(
        augment=False,
        add_noise=(input_noise_dataset > 0.0 and noise_on_test),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
    )

    if dataset == "cifar10":
        train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tfm)
        val_full = datasets.CIFAR10(root=data_root, train=True, download=False, transform=val_tfm)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfm)
        num_classes = 10
    else:
        train_full = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_tfm)
        val_full = datasets.CIFAR100(root=data_root, train=True, download=False, transform=val_tfm)
        test_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_tfm)
        num_classes = 100

    n = len(train_full)
    tr_idx, va_idx = split_train_val(n, val_frac, seed)

    train_ds = Subset(train_full, tr_idx.tolist())
    val_ds = Subset(val_full, va_idx.tolist()) if va_idx.numel() > 0 else None

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if val_ds is not None else None
    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    x_test, y_test = dataset_to_tensors(test_ds)

    if flatten:
        in_dim: Optional[int] = int(x_test.shape[1])
        input_shape: Optional[Tuple[int, ...]] = None
    else:
        in_dim = None
        input_shape = (3, 32, 32)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        in_dim=in_dim,
        input_shape=input_shape,
        x_test=x_test,
        y_test=y_test,
    )


@register_dataset("cifar10")
def make_cifar10_dataset(**kwargs) -> DataBundle:
    return _make_cifar_dataset(dataset="cifar10", **kwargs)


@register_dataset("cifar100")
def make_cifar100_dataset(**kwargs) -> DataBundle:
    return _make_cifar_dataset(dataset="cifar100", **kwargs)
