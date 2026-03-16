"""Convenience helpers for building configured callback stacks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .registry import CallbackContext, build_callback, get_callback_names


def build_configured_callbacks(
    *,
    args: Any,
    mode: str,
    dataset: str | None,
    model: Any,
    optimizer: Any,
    schedulers: list[Any],
) -> list[Any]:
    callback_specs = list(getattr(args, "callbacks", []) or [])
    if not callback_specs:
        return []

    available = set(get_callback_names())
    out: list[Any] = []
    for idx, spec in enumerate(callback_specs):
        if not isinstance(spec, Mapping):
            raise ValueError(f"callbacks[{idx}] must be a table/object.")

        name = str(spec.get("name", "")).strip().lower()
        if not name:
            raise ValueError(f"callbacks[{idx}].name must be non-empty.")
        if not bool(spec.get("enabled", True)):
            continue
        if name not in available:
            raise ValueError(f"Unknown callback '{name}'. Available: {sorted(available)}")

        params_raw = spec.get("params", {})
        if not isinstance(params_raw, Mapping):
            raise ValueError(f"callbacks[{idx}].params must be a table/object.")

        ctx = CallbackContext(
            args=args,
            mode=mode,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            schedulers=schedulers,
            params=dict(params_raw),
            index=idx,
            extra={"callback_spec": dict(spec)},
        )
        out.append(build_callback(name, ctx))
    return out
