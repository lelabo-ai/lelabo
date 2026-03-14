from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
import torch.nn as nn

from ..core.activations import (
    cache_pair_activation_module_types,
    cache_pair_activation_name_from_module,
)
from .blocks import BlockSpec, ResolvedBlock, normalize_standard_cache
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
    target_view: Literal["declared", "execution", "paired_execution"] = "execution"
    trainable_module_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d)
    observed_module_types: tuple[type[nn.Module], ...] = ()
    observed_module_names: tuple[str, ...] = ()
    capture_inputs: bool = True
    capture_outputs: bool = True
    require_single_call: bool = False
    require_single_output_head: bool = False
    require_input_ndim: int | None = None
    require_output_ndim: int | None = None


@dataclass(frozen=True)
class _CapturePlan:
    capture_inputs: bool
    capture_outputs: bool
    capture_all_calls: bool
    include_steps: bool
    needs_pairing: bool


@dataclass(frozen=True)
class _DeclaredBlocksResolution:
    status: Literal["absent", "ok", "invalid", "error"]
    specs: tuple[BlockSpec, ...] = ()
    detail: str = ""


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


def _normalize_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_standard_cache(dict(cache))


def _normalize_target_view(raw: Any) -> Literal["declared", "execution", "paired_execution"]:
    value = str(raw).strip().lower()
    if value not in {"declared", "execution", "paired_execution"}:
        raise ValueError(
            f"Cache provider: unsupported target_view '{raw}'. "
            "Expected one of: declared, execution, paired_execution."
        )
    return value


def _capture_plan(spec: CacheSpec) -> _CapturePlan:
    target_view = _normalize_target_view(spec.target_view)
    needs_pairing = target_view == "paired_execution"
    capture_inputs = bool(spec.capture_inputs or spec.require_input_ndim is not None)
    capture_outputs = bool(
        spec.capture_outputs
        or spec.require_output_ndim is not None
        or needs_pairing
    )
    return _CapturePlan(
        capture_inputs=capture_inputs,
        capture_outputs=capture_outputs,
        capture_all_calls=bool(spec.require_single_call),
        include_steps=True,
        needs_pairing=needs_pairing,
    )


def _resolve_declared_blocks(model: nn.Module) -> _DeclaredBlocksResolution:
    if not hasattr(model, "declare_blocks"):
        return _DeclaredBlocksResolution(status="absent")

    getter = getattr(model, "declare_blocks", None)
    if not callable(getter):
        return _DeclaredBlocksResolution(
            status="invalid",
            detail="attribute 'declare_blocks' exists but is not callable.",
        )

    try:
        raw = getter()
    except Exception as exc:
        return _DeclaredBlocksResolution(
            status="error",
            detail=f"{exc.__class__.__name__}: {exc}",
        )

    if not isinstance(raw, list):
        return _DeclaredBlocksResolution(
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
        return _DeclaredBlocksResolution(
            status="invalid",
            detail=(
                "returned list contains invalid block entries "
                f"({invalid_count} invalid out of {len(raw)})."
            ),
        )
    return _DeclaredBlocksResolution(status="ok", specs=tuple(out))


def _raise_declared_blocks_error(resolution: _DeclaredBlocksResolution) -> None:
    if resolution.status == "invalid":
        raise ValueError(
            "Cache provider: model.declare_blocks() returned invalid data "
            f"({resolution.detail})"
        )
    if resolution.status == "error":
        raise ValueError(
            "Cache provider: model.declare_blocks() raised: "
            f"{resolution.detail}"
        )


def declares_blocks(model: nn.Module) -> bool:
    resolution = _resolve_declared_blocks(model)
    if resolution.status == "absent":
        return False
    _raise_declared_blocks_error(resolution)
    return bool(resolution.specs)


def resolve_declared_blocks(model: nn.Module) -> list[BlockSpec]:
    resolution = _resolve_declared_blocks(model)
    if resolution.status == "absent":
        return []
    _raise_declared_blocks_error(resolution)
    return list(resolution.specs)


def _named_module_specs(model: nn.Module) -> list[BlockSpec]:
    specs: list[BlockSpec] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if any(True for _ in module.children()):
            continue
        specs.append(
            BlockSpec(
                name=name,
                module=module,
                rep="identity",
                is_output=False,
                group="main",
            )
        )
    return specs


def _select_auto_block_specs(model: nn.Module, spec: CacheSpec) -> list[BlockSpec]:
    named = _named_module_specs(model)
    by_name_all = {str(item.name): item for item in named}

    observed_names = set(_as_strings(spec.observed_module_names))
    observed_types = _as_types(spec.observed_module_types)

    candidate_pool: list[BlockSpec] = []
    seen_pool_ids: set[int] = set()
    for candidate in named:
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
            if str(candidate.name) not in observed_names:
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


def _pairing_capture_specs(model: nn.Module) -> list[BlockSpec]:
    return _named_module_specs(model)


def _merge_capture_specs(*parts: Sequence[BlockSpec]) -> list[BlockSpec]:
    out: list[BlockSpec] = []
    seen_mod_ids: set[int] = set()
    for part in parts:
        for spec in part:
            mid = id(spec.module)
            if mid in seen_mod_ids:
                continue
            seen_mod_ids.add(mid)
            out.append(spec)
    return out


def _runtime_block_specs(cache: Mapping[str, Any], fallback: Sequence[BlockSpec]) -> list[BlockSpec]:
    runtime = cache.get("_runtime")
    if not isinstance(runtime, Mapping):
        runtime = {}
    raw_specs = runtime.get("block_specs_runtime")
    if isinstance(raw_specs, list) and raw_specs:
        out: list[BlockSpec] = []
        for item in raw_specs:
            spec = _as_block_spec(item)
            if spec is not None:
                out.append(spec)
        if out:
            return out
    return list(fallback)


def _cache_maps(
    cache: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], list[dict[str, Any]]]:
    module_inputs = cache.get("module_inputs", {})
    module_outputs = cache.get("module_outputs", {})
    call_count = cache.get("call_count_by_name", {})
    runtime = cache.get("_runtime", {})
    steps = runtime.get("steps", []) if isinstance(runtime, Mapping) else []

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


def _capture_name_by_module_id(specs: Sequence[BlockSpec]) -> dict[int, str]:
    return {id(spec.module): str(spec.name) for spec in specs}


def _actual_spec_by_module_id(specs: Sequence[BlockSpec]) -> dict[int, BlockSpec]:
    return {id(spec.module): spec for spec in specs}


def _make_resolved_block(
    *,
    name: str,
    module: nn.Module,
    spec: BlockSpec | None,
    is_trainable: bool,
    is_output: bool,
    x: Any,
    u: Any,
    call_count: int,
) -> ResolvedBlock:
    return ResolvedBlock(
        name=str(name),
        module=module,
        spec=spec,
        rep=str(spec.rep if spec is not None else "identity"),
        group=str(spec.group if spec is not None else "main"),
        is_output=bool(is_output),
        is_trainable=bool(is_trainable),
        x=x,
        u=u,
        h=None,
        activation_name=None,
        exec_module=None,
        exec_span_names=(),
        call_count=int(call_count),
    )


def _build_execution_blocks(
    cache: Mapping[str, Any],
    auto_specs: Sequence[BlockSpec],
    capture_specs: Sequence[BlockSpec],
    spec: CacheSpec,
    declared_specs: Sequence[BlockSpec],
) -> list[ResolvedBlock]:
    module_inputs, module_outputs, call_count, steps = _cache_maps(cache)
    trainable_types = _as_types(spec.trainable_module_types)
    capture_name_by_mid = _capture_name_by_module_id(capture_specs)
    capture_spec_by_mid = _actual_spec_by_module_id(capture_specs)

    selected_names_by_mid = {
        id(item.module): capture_name_by_mid.get(id(item.module), str(item.name))
        for item in auto_specs
    }
    selected_name_set = set(selected_names_by_mid.values())

    ordered_names: list[str] = []
    seen: set[str] = set()
    for step in steps:
        name = str(step.get("name", ""))
        if not name or name in seen or name not in selected_name_set:
            continue
        seen.add(name)
        ordered_names.append(name)

    for item in auto_specs:
        name = selected_names_by_mid[id(item.module)]
        if name in seen:
            continue
        seen.add(name)
        ordered_names.append(name)

    by_name: dict[str, BlockSpec] = {}
    for item in auto_specs:
        actual = capture_spec_by_mid.get(id(item.module), item)
        by_name[str(actual.name)] = actual

    declared_output_names = {str(item.name) for item in declared_specs if bool(item.is_output)}
    has_declared_output = bool(declared_output_names.intersection(by_name))
    fallback_output_idx = -1

    execution: list[ResolvedBlock] = []
    for i, name in enumerate(ordered_names):
        block_spec = by_name[name]
        module = block_spec.module
        is_trainable = bool(trainable_types and isinstance(module, trainable_types))
        if is_trainable:
            fallback_output_idx = i
        execution.append(
            _make_resolved_block(
                name=name,
                module=module,
                spec=block_spec,
                is_trainable=is_trainable,
                is_output=bool(name in declared_output_names),
                x=module_inputs.get(name),
                u=module_outputs.get(name),
                call_count=int(call_count.get(name, 0)),
            )
        )

    if execution and not has_declared_output:
        idx = fallback_output_idx if fallback_output_idx >= 0 else (len(execution) - 1)
        execution[idx] = replace(execution[idx], is_output=True)

    return execution


def _build_declared_blocks(
    cache: Mapping[str, Any],
    declared_specs: Sequence[BlockSpec],
    spec: CacheSpec,
) -> list[ResolvedBlock]:
    if not declared_specs:
        return []

    module_inputs, module_outputs, call_count, _steps = _cache_maps(cache)
    trainable_types = _as_types(spec.trainable_module_types)

    out: list[ResolvedBlock] = []
    for block in declared_specs:
        out.append(
            _make_resolved_block(
                name=str(block.name),
                module=block.module,
                spec=block,
                is_trainable=bool(trainable_types and isinstance(block.module, trainable_types)),
                is_output=bool(block.is_output),
                x=module_inputs.get(block.name),
                u=module_outputs.get(block.name),
                call_count=int(call_count.get(block.name, 0)),
            )
        )
    return out


def _build_paired_execution_blocks(
    execution_blocks: Sequence[ResolvedBlock],
    capture_specs: Sequence[BlockSpec],
    cache: Mapping[str, Any],
) -> list[ResolvedBlock]:
    module_inputs, _module_outputs, _call_count, steps = _cache_maps(cache)
    if not steps:
        return list(execution_blocks)

    by_name = {str(item.name): item for item in capture_specs}
    execution_trainable_module_ids = {id(block.module) for block in execution_blocks if block.is_trainable}
    first_step_idx_by_name: dict[str, int] = {}
    for i, step in enumerate(steps):
        name = str(step.get("name", ""))
        if not name or name in first_step_idx_by_name:
            continue
        first_step_idx_by_name[name] = i

    out: list[ResolvedBlock] = list(execution_blocks)
    for idx, block in enumerate(out):
        if not block.is_trainable or block.is_output:
            continue

        step_idx = first_step_idx_by_name.get(block.name)
        if step_idx is None:
            continue

        out_ref = steps[step_idx].get("out_ref_id")
        if out_ref is None:
            continue

        current_ref = out_ref
        search_start = step_idx + 1
        matched_tensor = None
        matched_name = None

        while current_ref is not None:
            matched_step: dict[str, Any] | None = None
            matched_spec: BlockSpec | None = None

            for j in range(search_start, len(steps)):
                nxt = steps[j]
                if nxt.get("in_ref_id") != current_ref:
                    continue
                nxt_name = str(nxt.get("name", ""))
                nxt_spec = by_name.get(nxt_name)
                if nxt_spec is None:
                    continue
                matched_step = nxt
                matched_spec = nxt_spec
                search_start = j + 1
                break

            if matched_step is None or matched_spec is None:
                break

            activation_name = cache_pair_activation_name_from_module(matched_spec.module)
            if activation_name is not None or isinstance(matched_spec.module, cache_pair_activation_module_types()):
                matched_tensor = matched_step.get("out")
                matched_name = activation_name
                break

            if id(matched_spec.module) in execution_trainable_module_ids and matched_spec.module is not block.module:
                break

            current_ref = matched_step.get("out_ref_id")

        if matched_tensor is None:
            continue

        if not torch.is_tensor(block.h):
            x_tensor = module_inputs.get(block.name)
            out[idx] = replace(out[idx], h=matched_tensor, activation_name=matched_name, x=x_tensor if x_tensor is not None else block.x)
        else:
            out[idx] = replace(out[idx], h=matched_tensor, activation_name=matched_name)
    return out


def _attach_exec_modules(blocks: Sequence[ResolvedBlock], *, declared_view: bool) -> list[ResolvedBlock]:
    if not blocks:
        return []

    if declared_view:
        return [
            replace(
                block,
                exec_module=block.module if isinstance(block.module, nn.Module) else None,
                exec_span_names=(block.name,),
            )
            for block in blocks
        ]

    out = list(blocks)
    trainable_indices = [i for i, block in enumerate(out) if block.is_trainable]
    for pos, start_idx in enumerate(trainable_indices):
        stop_idx = trainable_indices[pos + 1] - 1 if pos + 1 < len(trainable_indices) else (len(out) - 1)
        if stop_idx < start_idx:
            continue

        segment = list(out[start_idx : stop_idx + 1])
        modules = [block.module for block in segment if isinstance(block.module, nn.Module)]
        if not modules:
            continue
        local_module = modules[0] if len(modules) == 1 else _LocalBlockChain(modules[0], modules[1:])
        out[start_idx] = replace(
            out[start_idx],
            exec_module=local_module,
            exec_span_names=tuple(block.name for block in segment),
        )
    return out


def _build_views(
    cache: Mapping[str, Any],
    auto_specs: Sequence[BlockSpec],
    capture_specs: Sequence[BlockSpec],
    spec: CacheSpec,
    declared_specs: Sequence[BlockSpec],
) -> dict[str, Any]:
    target_view = _normalize_target_view(spec.target_view)
    if target_view == "declared":
        return {
            "declared": _attach_exec_modules(
                _build_declared_blocks(cache, declared_specs, spec),
                declared_view=True,
            )
        }

    execution = _attach_exec_modules(
        _build_execution_blocks(cache, auto_specs, capture_specs, spec, declared_specs),
        declared_view=False,
    )
    if target_view == "execution":
        return {"execution": execution}

    paired_execution = _attach_exec_modules(
        _build_paired_execution_blocks(execution, capture_specs, cache),
        declared_view=False,
    )
    return {"paired_execution": paired_execution}


def _normalize_external_cache(cache: Mapping[str, Any] | None) -> dict[str, Any]:
    module_inputs = {}
    module_outputs = {}
    merged: dict[str, Any] = {}

    if isinstance(cache, Mapping):
        raw_inputs = cache.get("module_inputs")
        raw_outputs = cache.get("module_outputs")
        if isinstance(raw_inputs, Mapping):
            module_inputs = dict(raw_inputs)
        if isinstance(raw_outputs, Mapping):
            module_outputs = dict(raw_outputs)
        passthrough_keys = {
            "module_inputs",
            "module_outputs",
            "module_inputs_all",
            "module_outputs_all",
            "call_count_by_name",
            "cache_version",
            "_runtime",
            "steps",
            "block_specs_runtime",
        }
        for key, value in cache.items():
            key_str = str(key)
            if key_str in passthrough_keys:
                continue
            merged[key_str] = value

    merged["module_inputs"] = module_inputs
    merged["module_outputs"] = module_outputs
    return normalize_standard_cache(merged)


def _validate_spec(cache: Mapping[str, Any], views: Mapping[str, Any], spec: CacheSpec) -> None:
    target_view = _normalize_target_view(spec.target_view)
    blocks = views.get(target_view, [])
    if not isinstance(blocks, list):
        blocks = []
    blocks = [entry for entry in blocks if isinstance(entry, ResolvedBlock)]

    output_blocks = [entry for entry in blocks if entry.is_output]
    if spec.require_single_output_head and len(output_blocks) != 1:
        names = [entry.name for entry in output_blocks]
        raise ContractError(
            "CacheSpec requires exactly one output head, but found "
            f"{len(output_blocks)}: {names}."
        )

    if spec.require_single_call:
        counts = cache.get("call_count_by_name", {})
        if not isinstance(counts, Mapping):
            raise ContractError("CacheSpec requires single-call validation, but call counts are missing.")
        bad: list[str] = []
        for block in blocks:
            if int(counts.get(block.name, 0)) != 1:
                bad.append(f"{block.name}:{int(counts.get(block.name, 0))}")
        if bad:
            raise ContractError(
                "CacheSpec requires each observed block to be called exactly once; "
                f"violations: {', '.join(bad)}"
            )

    if spec.require_input_ndim is not None:
        exp = int(spec.require_input_ndim)
        bad: list[str] = []
        for block in blocks:
            if not block.is_trainable:
                continue
            if torch.is_tensor(block.x) and block.x.dim() == exp:
                continue
            bad.append(f"{block.name}:{getattr(block.x, 'shape', None)}")
        if bad:
            raise ContractError(
                f"CacheSpec requires input ndim={exp} for trainable blocks. Offending blocks: "
                + ", ".join(bad)
            )

    if spec.require_output_ndim is not None:
        exp = int(spec.require_output_ndim)
        bad: list[str] = []
        for block in blocks:
            if not block.is_trainable:
                continue
            if torch.is_tensor(block.u) and block.u.dim() == exp:
                continue
            bad.append(f"{block.name}:{getattr(block.u, 'shape', None)}")
        if bad:
            raise ContractError(
                f"CacheSpec requires output ndim={exp} for trainable blocks. Offending blocks: "
                + ", ".join(bad)
            )

    if target_view == "paired_execution":
        missing = [
            block.name
            for block in blocks
            if block.is_trainable and not block.is_output and not torch.is_tensor(block.h)
        ]
        if missing:
            raise ContractError(
                "Post-activation pairing is required but missing for blocks: "
                + ", ".join(missing)
                + ". If your model uses functional activations (torch.nn.functional.*), replace them "
                + "with nn.Module activations or register custom pairing support."
            )


def forward_with_standard_cache(model, *args, cache_spec: CacheSpec | None = None, **kwargs):
    spec = cache_spec or CacheSpec()
    plan = _capture_plan(spec)
    target_view = _normalize_target_view(spec.target_view)

    if isinstance(model, ModelCacheWrapper):
        wrapper = model
        source_model = wrapper.model
        declared_resolution = _resolve_declared_blocks(source_model)
        _raise_declared_blocks_error(declared_resolution)
        declared_specs = list(declared_resolution.specs)
        auto_specs = wrapper.declare_blocks() if target_view in {"execution", "paired_execution"} else []
        capture_specs = auto_specs
    else:
        source_model = model
        declared_resolution = _resolve_declared_blocks(source_model)
        _raise_declared_blocks_error(declared_resolution)
        declared_specs = list(declared_resolution.specs)
        auto_specs: list[BlockSpec] = []
        if target_view in {"execution", "paired_execution"}:
            auto_specs = _select_auto_block_specs(model, spec)

        pairing_specs = _pairing_capture_specs(model) if plan.needs_pairing else []
        if target_view == "declared":
            capture_specs = list(declared_specs)
        elif target_view == "execution":
            capture_specs = list(auto_specs)
        else:
            capture_specs = _merge_capture_specs(auto_specs, pairing_specs)

        if not capture_specs:
            model_res = model(*args, **kwargs)
            if (
                isinstance(model_res, tuple)
                and len(model_res) == 2
                and isinstance(model_res[1], Mapping)
            ):
                out = model_res[0]
                cache = _normalize_external_cache(model_res[1])
            else:
                out = model_res
                cache = normalize_standard_cache({"module_inputs": {}, "module_outputs": {}})
            views = _build_views(cache, auto_specs, capture_specs, spec, declared_specs)
            _validate_spec(cache, views, spec)
            return out, cache, views

        wrapper = ModelCacheWrapper(
            model,
            block_specs=capture_specs,
            capture_inputs=plan.capture_inputs,
            capture_outputs=plan.capture_outputs,
            capture_all_calls=plan.capture_all_calls,
            include_steps=plan.include_steps,
        )

    out, cache_raw = wrapper(*args, return_cache=True, **kwargs)
    cache = _normalize_cache(cache_raw)
    runtime_capture_specs = _runtime_block_specs(cache, capture_specs)
    views = _build_views(cache, auto_specs, runtime_capture_specs, spec, declared_specs)
    _validate_spec(cache, views, spec)
    return out, cache, views
