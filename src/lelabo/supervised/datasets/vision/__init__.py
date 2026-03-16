"""Vision dataset builders bundled with LeLabo."""

from __future__ import annotations


def make_mnist_dataset(*args, **kwargs):
    from .mnist import make_mnist_dataset as _impl

    return _impl(*args, **kwargs)


def make_cifar10_dataset(*args, **kwargs):
    from .cifar import make_cifar10_dataset as _impl

    return _impl(*args, **kwargs)


def make_cifar100_dataset(*args, **kwargs):
    from .cifar import make_cifar100_dataset as _impl

    return _impl(*args, **kwargs)


def make_ssdd_dataset(*args, **kwargs):
    from .ssdd import make_ssdd_dataset as _impl

    return _impl(*args, **kwargs)


def make_hrsid_dataset(*args, **kwargs):
    from .hrsid import make_hrsid_dataset as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "make_mnist_dataset",
    "make_cifar10_dataset",
    "make_cifar100_dataset",
    "make_ssdd_dataset",
    "make_hrsid_dataset",
]
"""Vision dataset builders bundled with LeLabo."""
