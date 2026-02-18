from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.registry import Registry


METRIC_REGISTRY = Registry("metrics", package="lab.metrics")


@dataclass
class MetricContext:
    args: Any
    mode: str = "supervised"
    dataset: str | None = None
    algo: str | None = None
    extra: dict[str, Any] | None = None


def register_metric(name: str):
    return METRIC_REGISTRY.register(name)


def build_metric(name: str, ctx: MetricContext):
    if not METRIC_REGISTRY.names():
        METRIC_REGISTRY.discover()
    builder = METRIC_REGISTRY.get(name)
    return builder(ctx)


def get_metric_names() -> list[str]:
    if not METRIC_REGISTRY.names():
        METRIC_REGISTRY.discover()
    return METRIC_REGISTRY.names()


def parse_metric_names(raw: Any) -> list[str]:
    if raw is None:
        return []

    chunks: list[str] = []
    if isinstance(raw, str):
        chunks = [raw]
    elif isinstance(raw, (list, tuple, set)):
        chunks = [str(item) for item in raw]
    else:
        chunks = [str(raw)]

    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for comma_part in str(chunk).split(","):
            for token in comma_part.strip().split():
                key = token.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(key)
    return out
