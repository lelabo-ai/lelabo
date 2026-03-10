from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
import warnings

import torch
import torch.nn as nn

from ..core.activations import CACHE_AUTO_PAIR_ACTIVATION_MODULE_TYPES
from .blocks import BlockSpec, normalize_standard_cache
from .cache_wrapper import ModelCacheWrapper


class ContractError(RuntimeError):
    """Raised when cache/model contract requirements are not satisfied."""


class _LocalBlockChain(nn.Module):
    """Callable chain used by local rules to replay reconstructed local blocks."""

    def __init__(self, root: nn.Module, post_modules: Sequence[nn.Module]):
        super().__init__()
        self.root = root
        self.post = nn.ModuleList(list(post_modules))

    def forward(self, x):
        y = self.root(x)
        for mod in self.post:
            y = mod(y)
        return y


@dataclass(frozen=True)
class CacheSpec:
    # --- Selection
    trainable_module_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d)
    observed_module_types: tuple[type[nn.Module], ...] = ()
    observed_module_names: tuple[str, ...] = ()
    declared_blocks_mode: Literal["ignore", "merge", "only"] = "merge"
    # --- Advanced capture controls
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_all_calls: bool = True
    capture_steps: bool = True
    # --- Constraints
    require_single_call: bool = False
    require_single_output_head: bool = False
    require_input_ndim: int | None = None
    require_output_ndim: int | None = None
    # --- Derived views / post-processing
    auto_pair_post_activation: bool = False
    auto_pair_activation_types: tuple[type[nn.Module], ...] = CACHE_AUTO_PAIR_ACTIVATION_MODULE_TYPES


# ============================================================
# Helpers
# ============================================================

def _as_types(raw: Sequence[type[nn.Module]] | type[nn.Module] | None) -> tuple[type[nn.Module], ...]:
    if raw is None:
        return ()
    if isinstance(raw, type):
        return (raw,)
    out: list[type[nn.Module]] = []
    for item in raw:
        if isinstance(item, type):
            out.append(item)
    return tuple(out)


def _as_strings(raw: Sequence[str] | str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        token = raw.strip()
        return (token,) if token else ()
    out: list[str] = []
    for item in raw:
        token = str(item).strip()
        if token:
            out.append(token)
    return tuple(out)


def _dedup_types(types: Sequence[type[nn.Module]]) -> tuple[type[nn.Module], ...]:
    seen: set[type[nn.Module]] = set()
    out: list[type[nn.Module]] = []
    for t in types:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return tuple(out)


def _as_block_spec(raw: Any) -> BlockSpec | None:
    name = getattr(raw, "name", None)
    module = getattr(raw, "module", None)
    if not isinstance(name, str) or not isinstance(module, nn.Module):
        return None

    rep = getattr(raw, "rep", "identity")
    group = getattr(raw, "group", "main")
    params = getattr(raw, "params", None)
    in_select = getattr(raw, "in_select", None)
    out_select = getattr(raw, "out_select", None)

    return BlockSpec(
        name=str(name),
        module=module,
        rep=str(rep),
        is_output=bool(getattr(raw, "is_output", False)),
        group=str(group),
        params=params,
        in_select=in_select,
        out_select=out_select,
    )


@dataclass(frozen=True)
class _GetBlocksResolution:
    status: Literal["absent", "ok", "invalid", "error"]
    specs: tuple[BlockSpec, ...] = ()
    detail: str = ""


def _resolve_get_blocks(model: nn.Module) -> _GetBlocksResolution:
    if not hasattr(model, "get_blocks"):
        return _GetBlocksResolution(status="absent")

    getter = getattr(model, "get_blocks", None)
    if not callable(getter):
        return _GetBlocksResolution(
            status="invalid",
            detail="attribute 'get_blocks' exists but is not callable.",
        )

    try:
        raw = getter()
    except Exception as exc:
        return _GetBlocksResolution(
            status="error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )

    if not isinstance(raw, list):
        return _GetBlocksResolution(
            status="invalid",
            detail=f"expected list, got {type(raw).__name__}.",
        )

    out: list[BlockSpec] = []
    invalid_count = 0
    for item in raw:
        spec = _as_block_spec(item)
        if spec is None:
            invalid_count += 1
            continue
        out.append(spec)

    if invalid_count > 0:
        return _GetBlocksResolution(
            status="invalid",
            detail=(
                "returned list contains invalid block entries "
                f"({invalid_count} invalid out of {len(raw)})."
            ),
        )
    return _GetBlocksResolution(status="ok", specs=tuple(out))


def _named_module_specs(model: nn.Module) -> list[BlockSpec]:
    specs: list[BlockSpec] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if any(True for _ in module.children()):
            continue
        specs.append(BlockSpec(name=name, module=module, rep="identity", is_output=False, group="main"))
    return specs


def _normalize_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_standard_cache(dict(cache))


def _normalize_declared_blocks_mode(raw: Any) -> Literal["ignore", "merge", "only"]:
    source = str(raw).strip().lower()
    if source not in {"ignore", "merge", "only"}:
        raise ValueError(
            f"Cache provider: unsupported declared_blocks_mode '{raw}'. "
            "Expected one of: ignore, merge, only."
        )
    return source


def _runtime_block_specs(cache: Mapping[str, Any], fallback: Sequence[BlockSpec]) -> list[BlockSpec]:
    runtime = cache.get("block_specs_runtime")
    if isinstance(runtime, list) and runtime:
        out: list[BlockSpec] = []
        for item in runtime:
            spec = _as_block_spec(item)
            if spec is not None:
                out.append(spec)
        if out:
            return out
    return list(fallback)


# ============================================================
# Selection
# ============================================================

def _raise_declared_blocks_only_error(resolution: _GetBlocksResolution) -> None:
    if resolution.status == "absent":
        raise ValueError(
            "Cache provider: declared_blocks_mode='only' requires model.get_blocks(), "
            "but model has no get_blocks() method."
        )
    if resolution.status == "error":
        raise ValueError(
            "Cache provider: declared_blocks_mode='only' failed because model.get_blocks() raised: "
            f"{resolution.detail}"
        )
    if resolution.status == "invalid":
        raise ValueError(
            "Cache provider: declared_blocks_mode='only' failed because model.get_blocks() "
            f"returned invalid data ({resolution.detail})"
        )
    raise ValueError(
        "Cache provider: declared_blocks_mode='only' requires model.get_blocks() "
        "to return a non-empty list of valid block specs (got empty list)."
    )


def _select_block_specs(
    model: nn.Module,
    spec: CacheSpec,
    declared: _GetBlocksResolution,
) -> list[BlockSpec]:
    declared_mode = _normalize_declared_blocks_mode(spec.declared_blocks_mode)
    named = _named_module_specs(model)
    explicit = list(declared.specs) if declared.status == "ok" else []

    if declared_mode == "only":
        if not explicit:
            _raise_declared_blocks_only_error(declared)
        candidate_sources = list(explicit)
    elif declared_mode == "ignore":
        candidate_sources = list(named)
    else:
        if declared.status in {"invalid", "error"}:
            warnings.warn(
                "Cache provider: declared_blocks_mode='merge' ignored model.get_blocks() "
                f"because it is {declared.status} ({declared.detail}). "
                "Falling back to auto-discovered blocks.",
                UserWarning,
                stacklevel=2,
            )
        candidate_sources = list(explicit) + list(named)

    by_name_all: dict[str, BlockSpec] = {}
    for source in candidate_sources:
        by_name_all[str(source.name)] = source

    observed_names = set(_as_strings(spec.observed_module_names))
    explicit_observed_types = _as_types(spec.observed_module_types)

    observed_types = explicit_observed_types
    if observed_types and spec.auto_pair_post_activation:
        observed_types = _dedup_types(observed_types + _as_types(spec.auto_pair_activation_types))

    candidate_pool: list[BlockSpec] = []
    seen_pool_ids: set[int] = set()
    for candidate in candidate_sources:
        mid = id(candidate.module)
        if mid in seen_pool_ids:
            continue
        seen_pool_ids.add(mid)
        candidate_pool.append(candidate)

    selected: list[BlockSpec] = []
    seen_mod_ids: set[int] = set()

    if observed_names:
        missing = [name for name in observed_names if name not in by_name_all]
        if missing:
            raise ValueError(
                "Cache provider: observed_module_names contain unknown module(s): "
                + ", ".join(sorted(missing))
            )

        for candidate in candidate_pool:
            name = str(candidate.name)
            if name not in observed_names:
                continue
            if observed_types and not isinstance(candidate.module, observed_types):
                continue
            mid = id(candidate.module)
            if mid in seen_mod_ids:
                continue
            seen_mod_ids.add(mid)
            selected.append(candidate)
    else:
        for candidate in candidate_pool:
            if observed_types and not isinstance(candidate.module, observed_types):
                continue
            mid = id(candidate.module)
            if mid in seen_mod_ids:
                continue
            seen_mod_ids.add(mid)
            selected.append(candidate)

    if not selected:
        raise ValueError("Cache provider: no modules selected for cache collection.")

    return selected


# ============================================================
# Views
# ============================================================

def _cache_maps(
    cache: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], list[dict[str, Any]]]:
    module_inputs = cache.get("module_inputs", {})
    module_outputs = cache.get("module_outputs", {})
    call_count = cache.get("call_count_by_name", {})
    steps = cache.get("steps", [])

    if not isinstance(module_inputs, Mapping):
        module_inputs = {}
    if not isinstance(module_outputs, Mapping):
        module_outputs = {}
    if not isinstance(call_count, Mapping):
        call_count = {}
    if not isinstance(steps, list):
        steps = []
    steps = [step for step in steps if isinstance(step, dict)]
    return module_inputs, module_outputs, call_count, steps


def _make_block_entry(
    *,
    name: str,
    module: nn.Module,
    rep: str,
    group: str,
    is_trainable: bool,
    is_output: bool,
    x: Any,
    u: Any,
    h: Any = None,
    activation_name: str | None = None,
    call_count: int = 0,
    segment_names: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "name": str(name),
        "module": module,
        "type": module.__class__.__name__,
        "rep": str(rep),
        "group": str(group),
        "is_trainable": bool(is_trainable),
        "is_output": bool(is_output),
        "x": x,
        "u": u,
        "h": h,
        "activation_name": activation_name,
        "call_count": int(call_count),
        "available": bool(torch.is_tensor(x) or torch.is_tensor(u)),
        "segment_names": tuple(str(item) for item in segment_names),
    }


def _build_execution_blocks(
    cache: Mapping[str, Any],
    selected_blocks: Sequence[BlockSpec],
    spec: CacheSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    module_inputs, module_outputs, call_count, steps = _cache_maps(cache)
    trainable_types = _as_types(spec.trainable_module_types)
    by_name = {str(b.name): b for b in selected_blocks}

    ordered_names: list[str] = []
    seen: set[str] = set()
    for step in steps:
        name = str(step.get("name", ""))
        if not name or name in seen:
            continue
        if name not in by_name:
            continue
        seen.add(name)
        ordered_names.append(name)

    for block in selected_blocks:
        name = str(block.name)
        if name in seen:
            continue
        seen.add(name)
        ordered_names.append(name)

    execution: list[dict[str, Any]] = []
    declared_output_names = {str(block.name) for block in selected_blocks if bool(block.is_output)}
    has_declared_output = bool(declared_output_names)
    fallback_output_idx = -1
    for i, name in enumerate(ordered_names):
        block = by_name[name]
        module = block.module
        is_trainable = bool(trainable_types and isinstance(module, trainable_types))
        if is_trainable:
            fallback_output_idx = i
        execution.append(
            _make_block_entry(
                name=name,
                module=module,
                rep=str(block.rep),
                group=str(block.group),
                is_trainable=is_trainable,
                is_output=bool(name in declared_output_names),
                x=module_inputs.get(name),
                u=module_outputs.get(name),
                call_count=int(call_count.get(name, 0)),
            )
        )

    if (not has_declared_output) and fallback_output_idx >= 0:
        execution[fallback_output_idx]["is_output"] = True

    output_blocks = [entry for entry in execution if bool(entry.get("is_output", False))]
    return execution, output_blocks


def _apply_auto_pair_post_activation(
    execution_blocks: Sequence[dict[str, Any]],
    selected_blocks: Sequence[BlockSpec],
    cache: Mapping[str, Any],
    spec: CacheSpec,
) -> None:
    if not spec.auto_pair_post_activation:
        return

    _module_inputs, _module_outputs, _call_count, steps = _cache_maps(cache)
    if not steps:
        return

    activation_types = _as_types(spec.auto_pair_activation_types)
    if not activation_types:
        return

    by_name = {str(b.name): b for b in selected_blocks}
    first_step_idx_by_name: dict[str, int] = {}
    for i, step in enumerate(steps):
        name = str(step.get("name", ""))
        if not name or name in first_step_idx_by_name:
            continue
        first_step_idx_by_name[name] = i

    trainable_types = _as_types(spec.trainable_module_types)

    for block in execution_blocks:
        if not bool(block.get("is_trainable", False)):
            continue
        if bool(block.get("is_output", False)):
            continue

        name = str(block.get("name", ""))
        step_idx = first_step_idx_by_name.get(name)
        if step_idx is None:
            continue

        out_ref = steps[step_idx].get("out_ref_id")
        if out_ref is None:
            continue

        current_ref = out_ref
        search_start = step_idx + 1
        while current_ref is not None:
            matched_step: dict[str, Any] | None = None
            matched_spec: BlockSpec | None = None

            for j in range(search_start, len(steps)):
                nxt = steps[j]
                if nxt.get("in_ref_id") != current_ref:
                    continue
                nxt_name = str(nxt.get("name", ""))
                nxt_block = by_name.get(nxt_name)
                if nxt_block is None:
                    continue
                matched_step = nxt
                matched_spec = nxt_block
                search_start = j + 1
                break

            if matched_step is None or matched_spec is None:
                break

            module = matched_spec.module
            if isinstance(module, activation_types):
                block["h"] = matched_step.get("out")
                block["activation_name"] = str(matched_step.get("name", ""))
                break

            if trainable_types and isinstance(module, trainable_types):
                break

            current_ref = matched_step.get("out_ref_id")


def _build_trainable_segments(execution_blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    trainable_indices = [
        i for i, block in enumerate(execution_blocks)
        if bool(block.get("is_trainable", False))
    ]
    if not trainable_indices:
        return []

    out: list[dict[str, Any]] = []
    for pos, start_idx in enumerate(trainable_indices):
        stop_idx = (
            trainable_indices[pos + 1] - 1
            if (pos + 1 < len(trainable_indices))
            else (len(execution_blocks) - 1)
        )
        if stop_idx < start_idx:
            continue

        segment = list(execution_blocks[start_idx : stop_idx + 1])
        segment_names = tuple(str(b.get("name", "")) for b in segment)
        modules = [b.get("module") for b in segment if isinstance(b.get("module"), nn.Module)]
        if not modules:
            continue

        if len(modules) == 1:
            local_module = modules[0]
        else:
            local_module = _LocalBlockChain(modules[0], modules[1:])

        root = segment[0]
        tail = segment[-1]
        out.append(
            _make_block_entry(
                name=str(root.get("name", "")),
                module=local_module,
                rep=str(root.get("rep", "identity")),
                group=str(root.get("group", "main")),
                is_trainable=True,
                is_output=any(bool(block.get("is_output", False)) for block in segment),
                x=root.get("x"),
                u=tail.get("u"),
                h=root.get("h"),
                activation_name=root.get("activation_name"),
                call_count=int(root.get("call_count", 0)),
                segment_names=segment_names,
            )
        )

    return out


def _build_declared_blocks(
    cache: Mapping[str, Any],
    declared: _GetBlocksResolution,
    spec: CacheSpec,
) -> list[dict[str, Any]]:
    if declared.status != "ok":
        return []

    specs = list(declared.specs)
    if not specs:
        return []

    module_inputs, module_outputs, call_count, _steps = _cache_maps(cache)
    trainable_types = _as_types(spec.trainable_module_types)

    out: list[dict[str, Any]] = []
    for block in specs:
        name = str(block.name)
        out.append(
            _make_block_entry(
                name=name,
                module=block.module,
                rep=str(block.rep),
                group=str(block.group),
                is_trainable=bool(trainable_types and isinstance(block.module, trainable_types)),
                is_output=bool(block.is_output),
                x=module_inputs.get(name),
                u=module_outputs.get(name),
                h=None,
                activation_name=None,
                call_count=int(call_count.get(name, 0)),
            )
        )
    return out


def _build_views(
    cache: Mapping[str, Any],
    selected_blocks: Sequence[BlockSpec],
    spec: CacheSpec,
    declared: _GetBlocksResolution,
) -> dict[str, Any]:
    execution_blocks, output_blocks = _build_execution_blocks(cache, selected_blocks, spec)
    _apply_auto_pair_post_activation(execution_blocks, selected_blocks, cache, spec)
    trainable_segments = _build_trainable_segments(execution_blocks)
    declared_blocks = _build_declared_blocks(cache, declared, spec)
    return {
        "execution_blocks": execution_blocks,
        "output_blocks": output_blocks,
        "declared_blocks": declared_blocks,
        "trainable_segments": trainable_segments,
    }


# ============================================================
# Validation
# ============================================================

def _validate_spec(cache: Mapping[str, Any], views: Mapping[str, Any], spec: CacheSpec) -> None:
    execution = views.get("execution_blocks", [])
    if not isinstance(execution, list):
        execution = []

    output_blocks = views.get("output_blocks", [])
    if not isinstance(output_blocks, list):
        output_blocks = []
    output_blocks = [entry for entry in output_blocks if isinstance(entry, Mapping)]

    if spec.require_single_output_head and len(output_blocks) != 1:
        names = [str(entry.get("name", "<unnamed>")) for entry in output_blocks]
        raise ContractError(
            "CacheSpec requires exactly one output head, but found "
            f"{len(output_blocks)}: {names}."
        )

    if spec.require_single_call:
        counts = cache.get("call_count_by_name", {})
        if not isinstance(counts, Mapping):
            raise ContractError("CacheSpec requires single-call validation, but call counts are missing.")
        bad: list[str] = []
        for block in execution:
            name = str(block.get("name", ""))
            if int(counts.get(name, 0)) != 1:
                bad.append(f"{name}:{int(counts.get(name, 0))}")
        if bad:
            raise ContractError(
                "CacheSpec requires each observed block to be called exactly once; "
                f"violations: {', '.join(bad)}"
            )

    if spec.require_input_ndim is not None:
        exp = int(spec.require_input_ndim)
        bad: list[str] = []
        for block in execution:
            if not bool(block.get("is_trainable", False)):
                continue
            x = block.get("x")
            if torch.is_tensor(x) and x.dim() == exp:
                continue
            bad.append(f"{block.get('name', '<unnamed>')}:{getattr(x, 'shape', None)}")
        if bad:
            raise ContractError(
                f"CacheSpec requires input ndim={exp} for trainable blocks. Offending blocks: "
                + ", ".join(bad)
            )

    if spec.require_output_ndim is not None:
        exp = int(spec.require_output_ndim)
        bad: list[str] = []
        for block in execution:
            if not bool(block.get("is_trainable", False)):
                continue
            u = block.get("u")
            if torch.is_tensor(u) and u.dim() == exp:
                continue
            bad.append(f"{block.get('name', '<unnamed>')}:{getattr(u, 'shape', None)}")
        if bad:
            raise ContractError(
                f"CacheSpec requires output ndim={exp} for trainable blocks. Offending blocks: "
                + ", ".join(bad)
            )

    if spec.auto_pair_post_activation:
        missing: list[str] = []
        for block in execution:
            if not bool(block.get("is_trainable", False)):
                continue
            if bool(block.get("is_output", False)):
                continue
            if not torch.is_tensor(block.get("h")):
                missing.append(str(block.get("name", "<unnamed>")))
        if missing:
            raise ContractError(
                "Post-activation pairing is required but missing for blocks: "
                + ", ".join(missing)
                + ". If your model uses functional activations (torch.nn.functional.*) or "
                "those activation modules are not observed, replace with nn.Module activations "
                "and/or include them in observed_module_types."
            )


# ============================================================
# Public API
# ============================================================

def forward_with_standard_cache(model, *args, cache_spec: CacheSpec | None = None, **kwargs):
    spec = cache_spec or CacheSpec()

    needs_inputs = bool(
        spec.capture_inputs
        or (spec.require_input_ndim is not None)
    )
    needs_outputs = bool(
        spec.capture_outputs
        or (spec.require_output_ndim is not None)
        or spec.auto_pair_post_activation
    )
    needs_all_calls = bool(spec.capture_all_calls or spec.require_single_call)
    needs_steps = bool(
        spec.capture_steps
        or spec.require_single_call
        or spec.auto_pair_post_activation
    )

    if isinstance(model, ModelCacheWrapper):
        wrapper = model
        selected_specs = wrapper.get_blocks()
        source_model = wrapper.model
        declared = _resolve_get_blocks(source_model)
    else:
        source_model = model
        declared = _resolve_get_blocks(source_model)
        selected_specs = _select_block_specs(model, spec, declared)
        wrapper = ModelCacheWrapper(
            model,
            block_specs=selected_specs,
            capture_inputs=needs_inputs,
            capture_outputs=needs_outputs,
            capture_all_calls=needs_all_calls,
            include_steps=needs_steps,
        )

    out, cache_raw = wrapper(*args, return_cache=True, **kwargs)
    cache = _normalize_cache(cache_raw)
    runtime_specs = _runtime_block_specs(cache, selected_specs)
    views = _build_views(cache, runtime_specs, spec, declared)
    _validate_spec(cache, views, spec)
    return out, cache, views
