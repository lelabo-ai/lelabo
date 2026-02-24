from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..capsule.registry import index_path
from ..core.registry import Registry
from ..core.utils.capsule_plugins import (
    find_active_capsule_root,
    load_capsule_plugins,
    load_installed_capsule_plugins,
    plugin_files_fingerprint,
    reset_capsule_plugin_cache,
)


METRIC_REGISTRY = Registry("metrics", package="lab.metrics")
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
    output_key: str | None = None,  # kept for API compatibility
    description: str | None = None,  # kept for API compatibility
    params: Mapping[str, str] | None = None,
):
    _ = output_key, description

    def _decorator(builder):
        registered = METRIC_REGISTRY.register(name)(builder)
        setattr(registered, "__metric_kind__", _normalize_kind(kind))
        setattr(registered, "__metric_params__", _normalize_param_keys(params))
        return registered

    return _decorator


def register_metric_fn(
    name: str,
    *,
    kind: str = "classification",
    fn: Callable[..., float] | None = None,
    output_key: str | None = None,
    description: str | None = None,  # kept for API compatibility
    params: Mapping[str, str] | None = None,
):
    """
    Register a metric from a single function.

    Supported kinds:
      - classification: fn(y_true, y_pred, metric_params) -> float
      - regression: fn(y_true, y_pred, metric_params) -> float
      - scalar: fn(value_sum, weight_sum, metric_params) -> float
    """
    _ = description

    def _decorate(user_fn: Callable[..., float]) -> Callable[..., float]:
        normalized_kind = _normalize_kind(kind)
        key_name = str(output_key or name)

        @register_metric(
            name,
            kind=normalized_kind,
            params=params,
        )
        def _builder(ctx: MetricContext):
            metric_params = ctx.metric_params(name)
            if normalized_kind == "classification":
                from .helpers import FunctionClassificationMetric

                return FunctionClassificationMetric(
                    output_key=key_name,
                    fn=user_fn,
                    metric_params=metric_params,
                )
            if normalized_kind == "regression":
                from .helpers import FunctionRegressionMetric

                return FunctionRegressionMetric(
                    output_key=key_name,
                    fn=user_fn,
                    metric_params=metric_params,
                )
            from .helpers import FunctionScalarMetric

            return FunctionScalarMetric(
                output_key=key_name,
                fn=user_fn,
                metric_params=metric_params,
            )

        _ = _builder
        return user_fn

    if fn is not None:
        return _decorate(fn)
    return _decorate


def _ensure_metric_baseline() -> None:
    global _BASE_METRIC_ITEMS
    if _BASE_METRIC_ITEMS is not None:
        return
    _BASE_METRIC_ITEMS = METRIC_REGISTRY.snapshot_discovered_items()


def _normalize_extra_roots(extra_capsule_roots: Sequence[Path] | None) -> tuple[Path, ...]:
    roots = {Path(root).resolve() for root in list(extra_capsule_roots or [])}
    return tuple(sorted(roots, key=str))


def _capsules_index_mtime_ns(capsules_dir: Path | None) -> int | None:
    idx = index_path(capsules_dir)
    try:
        return int(idx.stat().st_mtime_ns)
    except OSError:
        return None


def _build_refresh_key(
    *,
    capsules_dir: Path | None,
    extra_capsule_roots: Sequence[Path] | None,
) -> tuple[Any, ...]:
    active_root = find_active_capsule_root()
    normalized_extra = _normalize_extra_roots(extra_capsule_roots)
    strict_plugins = os.getenv("LELABO_STRICT_PLUGINS", "").strip().lower()
    return (
        str(index_path(capsules_dir).parent.resolve()),
        _capsules_index_mtime_ns(capsules_dir),
        str(active_root) if active_root is not None else None,
        plugin_files_fingerprint(active_root, kinds=("metrics",)),
        tuple(
            (str(root), plugin_files_fingerprint(root, kinds=("metrics",)))
            for root in normalized_extra
        ),
        strict_plugins,
    )


def _refresh_metric_registry(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> None:
    global _LAST_METRIC_REFRESH_KEY, _LAST_METRIC_ITEMS
    _ensure_metric_baseline()
    refresh_key = _build_refresh_key(
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    )
    if _LAST_METRIC_REFRESH_KEY == refresh_key and _LAST_METRIC_ITEMS is not None:
        METRIC_REGISTRY._items = dict(_LAST_METRIC_ITEMS)
        return

    METRIC_REGISTRY._items = dict(_BASE_METRIC_ITEMS or {})
    reset_capsule_plugin_cache()
    load_capsule_plugins(kinds=("metrics",))
    for root in _normalize_extra_roots(extra_capsule_roots):
        load_capsule_plugins(kinds=("metrics",), capsule_root=Path(root).resolve())
    load_installed_capsule_plugins(kinds=("metrics",), capsules_dir=capsules_dir)
    _LAST_METRIC_REFRESH_KEY = refresh_key
    _LAST_METRIC_ITEMS = dict(METRIC_REGISTRY._items)


def build_metric(name: str, ctx: MetricContext):
    _refresh_metric_registry()
    builder = METRIC_REGISTRY.get(name)
    return builder(ctx)


def get_metric_names(
    *,
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[str]:
    _refresh_metric_registry(capsules_dir=capsules_dir, extra_capsule_roots=extra_capsule_roots)
    return METRIC_REGISTRY.names()


def validate_metric_requests(
    metric_names: Sequence[str],
    *,
    ctx: MetricContext,
    task_kind: str | None,
) -> None:
    normalized_task = None if task_kind is None else str(task_kind).strip().lower()
    if normalized_task not in {None, "classification", "regression"}:
        raise ValueError("task_kind must be one of: classification, regression, or None.")

    _refresh_metric_registry()
    for name in metric_names:
        key = str(name).strip().lower()
        builder = METRIC_REGISTRY._items.get(key)
        if builder is None:
            continue

        metric_kind = str(getattr(builder, "__metric_kind__", "any")).strip().lower()
        if (
            normalized_task is not None
            and metric_kind in {"classification", "regression"}
            and metric_kind != normalized_task
        ):
            raise ValueError(
                f"Metric '{key}' is for {metric_kind}, but task is {normalized_task}."
            )

        declared = getattr(builder, "__metric_params__", None)
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
