"""Builtin update rule registrations."""

from __future__ import annotations

from .backprop import Backprop, Backpropagation
from .dfa import DirectFeedbackAlignment
from .dni import DNI
from .drtp import DirectRandomTargetProjection
from .fa import FeedbackAlignment
from .scl import SoftContrastiveLearning
from .softhebb import SoftHebb


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


def build_dni(ctx):
    from .builders import build_dni as _build_dni

    return _build_dni(ctx)


def build_scl(ctx):
    from .builders import build_scl as _build_scl

    return _build_scl(ctx)


def build_softhebb(ctx):
    from .builders import build_softhebb as _build_softhebb

    return _build_softhebb(ctx)


__all__ = [
    "Backpropagation",
    "Backprop",
    "DirectFeedbackAlignment",
    "DNI",
    "FeedbackAlignment",
    "DirectRandomTargetProjection",
    "SoftContrastiveLearning",
    "SoftHebb",
    "build_backprop",
    "build_dfa",
    "build_dni",
    "build_fa",
    "build_drtp",
    "build_scl",
    "build_softhebb",
]
"""Builtin update rule registrations."""
