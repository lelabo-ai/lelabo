from __future__ import annotations


def make_glue_dataset(*args, **kwargs):
    from .glue import make_glue_dataset as _impl

    return _impl(*args, **kwargs)


__all__ = ["make_glue_dataset"]
