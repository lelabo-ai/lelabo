from __future__ import annotations

from .backprop import Backprop, Backpropagation
from .dfa import DirectFeedbackAlignment


def build_backprop(ctx):
    from .builders import build_backprop as _build_backprop

    return _build_backprop(ctx)


def build_dfa(ctx):
    from .builders import build_dfa as _build_dfa

    return _build_dfa(ctx)


__all__ = [
    "Backpropagation",
    "Backprop",
    "DirectFeedbackAlignment",
    "build_backprop",
    "build_dfa",
]
