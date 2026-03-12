from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..core.registry import Registry
from ..capsule.plugins.snapshot import build_capsule_registry_snapshot


METRIC_REGISTRY = Registry("metrics", package="lelabo.metrics")
_BASE_METRIC_ITEMS: dict[str, Any] | None = None
_LAST_METRIC_REFRESH_KEY: tuple[Any, ...] | None = None
_LAST_METRIC_ITEMS: dict[str, Any] | None = None

_ALLOWED_KINDS = {"any", "classification", "regression", "scalar"}


@dataclass
class MetricContext:
    args: Any
    mode: str = "supervised"
    dataset: str | None = None
    algo: str | None = None
    extra: dict[str, Any] | None = None

    def metric_params(self, name: str, *aliases: str) -> dict[str, Any]:
        extra = self.extra if isinstance(self.extra, dict) else {}
        all_params = extra.get("metric_params", {})
        if not isinstance(all_params, dict):
            return {}
        for candidate in (str(name), *[str(x) for x in aliases]):
            node = all_params.get(candidate)
            if isinstance(node, Mapping):
                return dict(node)
        return {}


def _normalize_kind(kind: str | None) -> str:
    raw = "any" if kind is None else str(kind).strip().lower()
    if raw not in _ALLOWED_KINDS:
        raise ValueError(f"Unsupported metric kind '{kind}'. Expected one of: {sorted(_ALLOWED_KINDS)}")
    return raw


def _normalize_param_keys(params: Mapping[str, str] | None) -> tuple[str, ...] | None:
    if params is None:
        return None
    keys = sorted({str(k).strip() for k in dict(params).keys() if str(k).strip()})
    return tuple(keys)


def register_metric(
    name: str,
    *,
    kind: str | None = None,
    params: Mapping[str, str] | None = None,
):
    def _decorator(builder):
        registered = METRIC_REGISTRY.register(name)(builder)
        setattr(registered, "__metric_kind__", _normalize_kind(kind))
        setattr(registered, "__metric_params__", _normalize_param_keys(params))
        return registered

    return _decorator


def _ensure_metric_baseline() -> None:
    global _BASE_METRIC_ITEMS
    if _BASE_METRIC_ITEMS is not None:
        return
    _BASE_METRIC_ITEMS = METRIC_REGISTRY.snapshot_discovered_items()

def _metric_snapshot(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> Any:
    _ensure_metric_baseline()
    return build_capsule_registry_snapshot(
        registry_name="metrics",
        kind="metrics",
        builtins=dict(_BASE_METRIC_ITEMS or {}),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )


def build_metric(name: str, ctx: MetricContext):
    builder = _metric_snapshot().get(name)
    return builder(ctx)


def get_metric_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    return _metric_snapshot(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots).names()


def validate_metric_requests(
    metric_names: Sequence[str],
    *,
    ctx: MetricContext,
    task_kind: str | None,
) -> None:
    normalized_task = None if task_kind is None else str(task_kind).strip().lower()
    if normalized_task not in {None, "classification", "regression"}:
        raise ValueError("task_kind must be one of: classification, regression, or None.")

    snapshot = _metric_snapshot()
    for name in metric_names:
        key = str(name).strip().lower()
        builder = snapshot.get_builtin(key)
        export = snapshot.get_export(key)
        if builder is None and export is None:
            continue

        if builder is not None:
            metric_kind = str(getattr(builder, "__metric_kind__", "any")).strip().lower()
            declared = getattr(builder, "__metric_params__", None)
        else:
            metadata = dict(export.metadata or {})
            metric_kind = str(metadata.get("metric_kind", "any")).strip().lower()
            declared_raw = metadata.get("metric_params", None)
            declared = tuple(declared_raw) if isinstance(declared_raw, list) else declared_raw

        if (
            normalized_task is not None
            and metric_kind in {"classification", "regression"}
            and metric_kind != normalized_task
        ):
            raise ValueError(
                f"Metric '{key}' is for {metric_kind}, but task is {normalized_task}."
            )

        if declared is None:
            continue
        allowed = set(str(x) for x in declared)
        metric_params = ctx.metric_params(key)
        unknown = sorted([p for p in metric_params.keys() if p not in allowed])
        if unknown:
            raise ValueError(
                f"Unknown params for metric '{key}': {unknown}. "
                f"Allowed: {sorted(allowed)}"
            )


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
