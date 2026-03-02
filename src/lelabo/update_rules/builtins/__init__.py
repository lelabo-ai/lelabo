from __future__ import annotations

from .backprop import Backprop, Backpropagation
from .dfa import DirectFeedbackAlignment
from .drtp import DirectRandomTargetProjection
from .fa import FeedbackAlignment


def build_backprop(ctx):
    from .builders import build_backprop as _build_backprop

    return _build_backprop(ctx)


def build_dfa(ctx):
    from .builders import build_dfa as _build_dfa

    return _build_dfa(ctx)


def build_fa(ctx):
    from .builders import build_fa as _build_fa

    return _build_fa(ctx)


def build_drtp(ctx):
    from .builders import build_drtp as _build_drtp

    return _build_drtp(ctx)


__all__ = [
    "Backpropagation",
    "Backprop",
    "DirectFeedbackAlignment",
    "FeedbackAlignment",
    "DirectRandomTargetProjection",
    "build_backprop",
    "build_dfa",
    "build_fa",
    "build_drtp",
]
