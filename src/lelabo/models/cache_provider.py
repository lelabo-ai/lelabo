from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from .blocks import BlockSpec, normalize_standard_cache
from .cache_wrapper import ModelCacheWrapper


class ContractError(RuntimeError):
    """Raised when cache/model contract requirements are not satisfied."""


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


@dataclass(frozen=True)
class CacheSpec:
    # --- What to observe
    observed_module_types: tuple[type[nn.Module], ...] = ()
    observed_module_names: tuple[str, ...] = ()
    observed_groups: tuple[str, ...] = ()
    param_module_types: tuple[type[nn.Module], ...] = ()
    activation_module_types: tuple[type[nn.Module], ...] = ()
    # --- What to capture
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_all_calls: bool = True
    capture_steps: bool = True
    # --- Constraints (v2)
    require_single_call: bool = False
    require_single_output_head: bool = False
    require_input_ndim: int | None = None
    require_output_ndim: int | None = None
    require_activation_pairing: bool = False
    # --- Backward-compat constraints (v1)
    require_block_inputs: bool = True
    require_block_outputs: bool = False
    require_single_output: bool = False
    require_linear_only: bool = False
    require_ndim2_inputs: bool = False


def _normalize_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_standard_cache(dict(cache))


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


def _maybe_get_blocks(model: nn.Module) -> list[BlockSpec]:
    if not hasattr(model, "get_blocks"):
        return []
    try:
        raw = model.get_blocks()
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[BlockSpec] = []
    for item in raw:
        spec = _as_block_spec(item)
        if spec is not None:
            out.append(spec)
    return out


def _named_module_specs(model: nn.Module) -> list[BlockSpec]:
    specs: list[BlockSpec] = []
    for name, module in model.named_modules():
        if not name:
            continue
        specs.append(BlockSpec(name=name, module=module, rep="identity", is_output=False))
    return specs


def _pick_output_index(specs: Sequence[BlockSpec], param_types: tuple[type[nn.Module], ...]) -> int:
    # Prefer an explicit "head" name when available.
    for i, spec in enumerate(specs):
        if str(spec.name) == "head":
            return i
    # Otherwise prefer the last param block.
    if param_types:
        for i in range(len(specs) - 1, -1, -1):
            if isinstance(specs[i].module, param_types):
                return i
    return len(specs) - 1


def _mark_single_output(specs: Sequence[BlockSpec], out_idx: int) -> list[BlockSpec]:
    out: list[BlockSpec] = []
    for i, spec in enumerate(specs):
        out.append(
            BlockSpec(
                name=spec.name,
                module=spec.module,
                rep=spec.rep,
                is_output=bool(i == out_idx),
                group=spec.group,
                params=spec.params,
                in_select=spec.in_select,
                out_select=spec.out_select,
            )
        )
    return out


def _select_block_specs(model: nn.Module, spec: CacheSpec) -> list[BlockSpec]:
    explicit = _maybe_get_blocks(model)
    named = _named_module_specs(model)

    observed_types = _as_types(spec.observed_module_types)
    param_types = _as_types(spec.param_module_types)
    act_types = _as_types(spec.activation_module_types)
    if spec.require_linear_only and not param_types:
        param_types = (nn.Linear,)

    observed_names = set(_as_strings(spec.observed_module_names))
    observed_groups = set(_as_strings(spec.observed_groups))

    # If rule did not request explicit filtering and model provides get_blocks(), keep model contract.
    no_explicit_filter = (
        not observed_types
        and not observed_names
        and not observed_groups
        and not param_types
        and not act_types
    )
    if explicit and no_explicit_filter:
        selected = list(explicit)
    else:
        effective_types = observed_types
        if not effective_types:
            effective_types = _dedup_types(param_types + act_types)

        pool = explicit if explicit else named
        selected = []
        seen_mod_ids: set[int] = set()
        for candidate in pool:
            name = str(candidate.name)
            module = candidate.module
            group = str(candidate.group)

            if observed_names and name not in observed_names:
                continue
            if observed_groups and group not in observed_groups:
                continue
            if effective_types and not isinstance(module, effective_types):
                continue
            if id(module) in seen_mod_ids:
                continue
            seen_mod_ids.add(id(module))
            selected.append(candidate)

        # Supplement from named_modules when explicit blocks do not cover requested selectors.
        if explicit and (observed_names or effective_types):
            by_id = {id(s.module) for s in selected}
            for candidate in named:
                if id(candidate.module) in by_id:
                    continue
                name = str(candidate.name)
                if observed_names and name not in observed_names:
                    continue
                if effective_types and not isinstance(candidate.module, effective_types):
                    continue
                selected.append(candidate)
                by_id.add(id(candidate.module))

    if not selected:
        # Legacy default when nothing was explicitly requested.
        for candidate in named:
            if isinstance(candidate.module, (nn.Linear, nn.Conv2d)):
                selected.append(candidate)
        if not selected:
            raise ValueError("Cache provider: no modules selected for cache collection.")

    if not any(bool(s.is_output) for s in selected):
        out_idx = _pick_output_index(selected, param_types=_dedup_types(param_types + (nn.Linear, nn.Conv2d)))
        selected = _mark_single_output(selected, out_idx)

    return selected


def _collect_blocks(model, cache: Mapping[str, Any]) -> list[BlockSpec]:
    runtime = cache.get("block_specs_runtime", None)
    if isinstance(runtime, list) and runtime:
        out: list[BlockSpec] = []
        for item in runtime:
            spec = _as_block_spec(item)
            if spec is not None:
                out.append(spec)
        if out:
            return out

    if hasattr(model, "get_blocks"):
        try:
            blocks = model.get_blocks()
            if isinstance(blocks, list):
                out: list[BlockSpec] = []
                for item in blocks:
                    spec = _as_block_spec(item)
                    if spec is not None:
                        out.append(spec)
                if out:
                    return out
        except Exception:
            pass
    return []


def _build_views(cache: Mapping[str, Any], blocks: list[BlockSpec], spec: CacheSpec) -> dict[str, Any]:
    module_inputs = cache.get("module_inputs", {})
    module_outputs = cache.get("module_outputs", {})
    steps = cache.get("steps", [])
    steps = steps if isinstance(steps, list) else []

    param_types = _as_types(spec.param_module_types)
    if spec.require_linear_only and not param_types:
        param_types = (nn.Linear,)
    if not param_types:
        param_types = (nn.Linear, nn.Conv2d)

    act_types = _as_types(spec.activation_module_types)

    selected_blocks = list(blocks)
    param_blocks = [b for b in selected_blocks if isinstance(b.module, param_types)]
    activation_blocks = [b for b in selected_blocks if act_types and isinstance(b.module, act_types)]

    output_blocks = [b for b in param_blocks if bool(b.is_output)]
    if not output_blocks:
        output_blocks = [b for b in selected_blocks if bool(b.is_output)]
    hidden_blocks = [b for b in param_blocks if b not in output_blocks]

    local_blocks: list[dict[str, Any]] = []
    for block in param_blocks:
        name = str(block.name)
        local_blocks.append(
            {
                "name": name,
                "module": block.module,
                "is_output": bool(block.is_output),
                "x": module_inputs.get(name),
                "u": module_outputs.get(name),
                "h": None,
                "activation_name": None,
            }
        )

    if (act_types or spec.require_activation_pairing) and steps:
        param_names = {str(b.name) for b in param_blocks}
        act_names: set[str] = set()
        if act_types:
            act_names = {str(b.name) for b in selected_blocks if isinstance(b.module, act_types)}
        by_name_local = {str(lb["name"]): lb for lb in local_blocks}

        for i, step in enumerate(steps):
            name = str(step.get("name", ""))
            if name not in param_names:
                continue
            local = by_name_local.get(name)
            if local is None:
                continue
            out_ref = step.get("out_ref_id")
            if out_ref is None:
                continue
            for j in range(i + 1, len(steps)):
                nxt = steps[j]
                nxt_name = str(nxt.get("name", ""))
                if act_names and nxt_name not in act_names:
                    continue
                if nxt.get("in_ref_id") != out_ref:
                    continue
                local["h"] = nxt.get("out")
                local["activation_name"] = nxt_name
                break

    return {
        "blocks": selected_blocks,
        "selected_blocks": selected_blocks,
        "param_blocks": param_blocks,
        "activation_blocks": activation_blocks,
        "output_blocks": output_blocks,
        "hidden_blocks": hidden_blocks,
        "local_blocks": local_blocks,
    }


def _validate_spec(cache: Mapping[str, Any], views: Mapping[str, Any], spec: CacheSpec) -> None:
    if spec.require_block_inputs and not isinstance(cache.get("block_inputs"), dict):
        raise ContractError("Cache missing required key: 'block_inputs'.")
    if spec.require_block_outputs and not isinstance(cache.get("block_outputs"), dict):
        raise ContractError("Cache missing required key: 'block_outputs'.")

    output_blocks = views.get("output_blocks", [])
    if spec.require_single_output or spec.require_single_output_head:
        n_out = int(len(output_blocks)) if isinstance(output_blocks, list) else 0
        if n_out != 1:
            raise ContractError(f"CacheSpec requires exactly one output block, got {n_out}.")

    selected_blocks = views.get("selected_blocks", [])
    if not isinstance(selected_blocks, list):
        selected_blocks = []

    if spec.require_linear_only:
        non_linear = [b.name for b in selected_blocks if not isinstance(getattr(b, "module", None), nn.Linear)]
        if non_linear:
            raise ContractError(
                "CacheSpec requires linear-only blocks, found unsupported blocks: "
                + ", ".join(str(n) for n in non_linear)
            )

    if spec.require_single_call:
        counts = cache.get("call_count_by_name", {})
        if not isinstance(counts, Mapping):
            raise ContractError("CacheSpec requires single-call validation, but call counts are missing.")
        bad: list[str] = []
        for block in selected_blocks:
            name = str(getattr(block, "name", ""))
            if int(counts.get(name, 0)) != 1:
                bad.append(f"{name}:{int(counts.get(name, 0))}")
        if bad:
            raise ContractError(
                "CacheSpec requires each selected block to be called exactly once; "
                f"violations: {', '.join(bad)}"
            )

    module_inputs = cache.get("module_inputs", {})
    module_outputs = cache.get("module_outputs", {})
    if not isinstance(module_inputs, Mapping):
        module_inputs = {}
    if not isinstance(module_outputs, Mapping):
        module_outputs = {}

    if spec.require_ndim2_inputs:
        bad = []
        for block in selected_blocks:
            name = str(getattr(block, "name", ""))
            value = module_inputs.get(name)
            if torch.is_tensor(value) and value.dim() == 2:
                continue
            bad.append(f"{name}:{getattr(value, 'shape', None)}")
        if bad:
            raise ContractError(
                "CacheSpec requires 2D inputs for selected blocks. Offending blocks: "
                + ", ".join(bad)
            )

    if spec.require_input_ndim is not None:
        exp = int(spec.require_input_ndim)
        bad = []
        for block in views.get("param_blocks", []):
            name = str(getattr(block, "name", ""))
            value = module_inputs.get(name)
            if torch.is_tensor(value) and value.dim() == exp:
                continue
            bad.append(f"{name}:{getattr(value, 'shape', None)}")
        if bad:
            raise ContractError(
                f"CacheSpec requires input ndim={exp} for param blocks. Offending blocks: "
                + ", ".join(bad)
            )

    if spec.require_output_ndim is not None:
        exp = int(spec.require_output_ndim)
        bad = []
        for block in views.get("param_blocks", []):
            name = str(getattr(block, "name", ""))
            value = module_outputs.get(name)
            if torch.is_tensor(value) and value.dim() == exp:
                continue
            bad.append(f"{name}:{getattr(value, 'shape', None)}")
        if bad:
            raise ContractError(
                f"CacheSpec requires output ndim={exp} for param blocks. Offending blocks: "
                + ", ".join(bad)
            )

    if spec.require_activation_pairing:
        missing: list[str] = []
        for local in views.get("local_blocks", []):
            if bool(local.get("is_output", False)):
                continue
            if local.get("h") is None:
                missing.append(str(local.get("name", "<unnamed>")))
        if missing:
            raise ContractError(
                "Activation pairing is required but missing for blocks: "
                + ", ".join(missing)
                + ". If your model uses functional activations (torch.nn.functional.*), "
                + "replace them with nn.Module activations (e.g. nn.ReLU/nn.Tanh) or disable "
                + "require_activation_pairing for this rule."
            )


def forward_with_standard_cache(model, *args, cache_spec: CacheSpec | None = None, **kwargs):
    spec = cache_spec or CacheSpec()

    needs_inputs = bool(
        spec.capture_inputs
        or spec.require_block_inputs
        or spec.require_ndim2_inputs
        or (spec.require_input_ndim is not None)
    )
    needs_outputs = bool(
        spec.capture_outputs
        or spec.require_block_outputs
        or (spec.require_output_ndim is not None)
        or spec.require_activation_pairing
    )
    needs_all_calls = bool(spec.capture_all_calls or spec.require_single_call)
    needs_steps = bool(spec.capture_steps or spec.require_activation_pairing)

    if isinstance(model, ModelCacheWrapper):
        wrapper = model
    else:
        selected_specs = _select_block_specs(model, spec)
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
    blocks = _collect_blocks(wrapper, cache)
    views = _build_views(cache, blocks, spec)
    _validate_spec(cache, views, spec)
    return out, cache, views
