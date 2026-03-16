"""Tabular dataset builders bundled with LeLabo."""

from __future__ import annotations


def make_iris_dataset(*args, **kwargs):
    from .iris import make_iris_dataset as _impl

    return _impl(*args, **kwargs)


def make_breast_cancer_dataset(*args, **kwargs):
    from .breast_cancer import make_breast_cancer_dataset as _impl

    return _impl(*args, **kwargs)


__all__ = ["make_iris_dataset", "make_breast_cancer_dataset"]
"""Tabular dataset builders bundled with LeLabo."""
