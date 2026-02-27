# lab/models/blocks.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Optional, Sequence, Literal

import torch
import torch.nn as nn


# -------------------------
# Block spec
# -------------------------

InSelect = Callable[[tuple[Any, ...], Optional[dict[str, Any]]], Any]
OutSelect = Callable[[Any], Any]


@dataclass(frozen=True)
class BlockSpec:
    """Describe a module block to enable modular/local update rules.

    - name: stable identifier (used in caches)
    - module: nn.Module that implements the block forward
    - rep: representation hint (metadata only; NOT automatically applied to cached tensors)
    - is_output: True for the final prediction head (best-effort; can be refined at runtime)
    - group: optional namespace
    - params: optional explicit parameter list
    - in_select: optional selector for module inputs (args, kwargs) -> value
    - out_select: optional selector for module output -> value
    """

    name: str
    module: nn.Module
    rep: str = "identity"
    is_output: bool = False
    group: str = "main"
    params: Optional[Sequence[nn.Parameter]] = None
    in_select: Optional[InSelect] = None
    out_select: Optional[OutSelect] = None

    def iter_params(self):
        if self.params is not None:
            yield from self.params
        else:
            yield from self.module.parameters()


# -------------------------
# LeModule: auto blocks + auto return_cache
# -------------------------

CacheCopyPolicy = Literal["none", "detach", "clone"]
OutputStrategy = Literal["last_spec", "last_executed"]


class LeModule(nn.Module):
    """Mixin with automatic block discovery + optional automatic return_cache."""

    # --- Default block discovery
    auto_block_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d)
    auto_block_filter: Callable[[str, nn.Module], bool] | None = None

    # If set, use this exact name as output head (stable).
    auto_output_block_name: str | None = None

    # If None, we can learn output from execution trace when return_cache=True.
    auto_output_strategy: OutputStrategy = "last_executed"

    # --- Cache policy defaults
    auto_cache_copy: CacheCopyPolicy = "detach"  # "clone" is safest but heavier
    auto_cache_to_cpu: bool = False
    auto_cache_include_steps: bool = True

    # Selective cloning to protect against in-place ops (e.g. ReLU(inplace=True)).
    # IMPORTANT: this is about later in-place ops mutating the cached tensor storage.
    auto_cache_clone_outputs_for_types: tuple[type[nn.Module], ...] = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    )
    auto_cache_clone_inputs: bool = False

    # -------------------------
    # Tensor extraction / packing
    # -------------------------

    @staticmethod
    def _extract_first_tensor(obj: Any) -> torch.Tensor | None:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (tuple, list)):
            for v in obj:
                t = LeModule._extract_first_tensor(v)
                if t is not None:
                    return t
            return None
        if isinstance(obj, dict):
            for v in obj.values():
                t = LeModule._extract_first_tensor(v)
                if t is not None:
                    return t
            return None
        # HuggingFace ModelOutput often has .to_tuple()
        if hasattr(obj, "to_tuple"):
            try:
                return LeModule._extract_first_tensor(obj.to_tuple())
            except Exception:
                return None
        return None

    def _pack_cache_tensor(
        self,
        value: Any,
        *,
        copy_policy: CacheCopyPolicy | None = None,
        clone: bool = False,
    ) -> torch.Tensor | None:
        t = self._extract_first_tensor(value)
        if t is None:
            return None

        pol = copy_policy or self.auto_cache_copy
        if pol == "none":
            pass
        elif pol == "detach":
            t = t.detach()
        elif pol == "clone":
            t = t.detach().clone()
        else:
            t = t.detach()

        # Extra safety clone (for in-place ops later in the graph)
        if clone and pol != "clone":
            # if pol == none => clone keeps grad tracking, but we don't want that for cache
            if pol == "none":
                t = t.detach().clone()
            else:
                t = t.clone()

        if self.auto_cache_to_cpu:
            t = t.cpu()

        return t

    # -------------------------
    # Block discovery
    # -------------------------

    def get_blocks(self) -> list[BlockSpec]:
        pred = self.auto_block_filter
        all_named = [(n, m) for n, m in self.named_modules() if n]

        specs: list[BlockSpec] = []

        # Standard type/filter selection
        for name, module in all_named:
            keep = bool(pred(name, module)) if pred is not None else isinstance(module, self.auto_block_types)
            if not keep:
                continue
            # rep is metadata only (do NOT apply automatically)
            rep = "gap" if isinstance(module, nn.Conv2d) else "identity"
            specs.append(BlockSpec(name=name, module=module, rep=rep, is_output=False))

        if not specs:
            return []

        # Determine which name is output (best-effort)
        # Priority: explicit auto_output_block_name > runtime learned name > fallback last spec
        runtime_name = getattr(self, "_lemodule_runtime_output_name", None)
        output_name = self.auto_output_block_name or (runtime_name if self.auto_output_strategy == "last_executed" else None)

        if output_name is not None:
            marked = False
            out_specs: list[BlockSpec] = []
            for s in specs:
                is_out = s.name == output_name
                marked = marked or is_out
                out_specs.append(
                    BlockSpec(
                        name=s.name,
                        module=s.module,
                        rep=s.rep,
                        is_output=is_out,
                        group=s.group,
                        params=s.params,
                        in_select=s.in_select,
                        out_select=s.out_select,
                    )
                )
            specs = out_specs
            if not marked:
                # fallback to last spec
                last = specs[-1]
                specs[-1] = BlockSpec(
                    name=last.name,
                    module=last.module,
                    rep=last.rep,
                    is_output=True,
                    group=last.group,
                    params=last.params,
                    in_select=last.in_select,
                    out_select=last.out_select,
                )
        else:
            # fallback: mark last spec as output
            last = specs[-1]
            specs[-1] = BlockSpec(
                name=last.name,
                module=last.module,
                rep=last.rep,
                is_output=True,
                group=last.group,
                params=last.params,
                in_select=last.in_select,
                out_select=last.out_select,
            )

        return specs

    @property
    def blocks(self) -> list[BlockSpec]:
        return list(self.get_blocks())

    # -------------------------
    # return_cache plumbing
    # -------------------------

    def _forward_accepts_return_cache(self) -> bool:
        cached = getattr(self, "_lemodule_forward_has_return_cache", None)
        if cached is not None:
            return bool(cached)
        try:
            sig = inspect.signature(type(self).forward)
            has_param = "return_cache" in sig.parameters
        except Exception:
            has_param = False
        setattr(self, "_lemodule_forward_has_return_cache", bool(has_param))
        return bool(has_param)

    @staticmethod
    def _parse_pre_hook_args(hook_args: tuple[Any, ...]) -> tuple[nn.Module, tuple[Any, ...], Optional[dict[str, Any]]]:
        # Possible signatures:
        # - (module, args)                           [old]
        # - (module, args, kwargs)                   [with_kwargs=True]
        if len(hook_args) == 2:
            module, args = hook_args
            return module, args, None
        if len(hook_args) == 3:
            module, args, kwargs = hook_args
            return module, args, kwargs
        raise TypeError(f"Unexpected forward_pre_hook signature: {len(hook_args)} args")

    @staticmethod
    def _parse_fwd_hook_args(hook_args: tuple[Any, ...]) -> tuple[nn.Module, tuple[Any, ...], Optional[dict[str, Any]], Any]:
        # Possible signatures:
        # - (module, args, output)                   [old]
        # - (module, args, kwargs, output)           [with_kwargs=True]
        if len(hook_args) == 3:
            module, args, output = hook_args
            return module, args, None, output
        if len(hook_args) == 4:
            module, args, kwargs, output = hook_args
            return module, args, kwargs, output
        raise TypeError(f"Unexpected forward_hook signature: {len(hook_args)} args")

    def _auto_forward_with_cache(self, *args, **kwargs):
        specs = list(self.get_blocks())
        if not specs:
            out = super().__call__(*args, **kwargs)
            cache = {"block_inputs": {}, "block_outputs": {}}
            if self.auto_cache_include_steps:
                cache["steps"] = []
            return out, cache

        # Enforce unique module instances in specs (mapping name <-> module must be well-defined)
        seen_ids: set[int] = set()
        for spec in specs:
            mid = id(spec.module)
            if mid in seen_ids:
                raise ValueError(
                    "LeModule automatic cache expects unique module instances in get_blocks(). "
                    "If you need two names for the same module, wrap it or use steps-based rules."
                )
            seen_ids.add(mid)

        # Compat dicts: last occurrence wins
        block_inputs: dict[str, torch.Tensor] = {}
        block_outputs: dict[str, torch.Tensor] = {}

        # Robust: keep all occurrences by name
        block_inputs_all: dict[str, list[torch.Tensor]] = defaultdict(list)
        block_outputs_all: dict[str, list[torch.Tensor]] = defaultdict(list)

        steps: list[dict[str, Any]] = []
        pending_by_module: dict[int, list[torch.Tensor | None]] = defaultdict(list)
        call_count_by_name: dict[str, int] = defaultdict(int)
        hooks = []

        # Hook registration with kwargs support if available
        def _register_pre_hook(mod: nn.Module, fn):
            try:
                return mod.register_forward_pre_hook(fn, with_kwargs=True)
            except TypeError:
                return mod.register_forward_pre_hook(fn)

        def _register_fwd_hook(mod: nn.Module, fn):
            try:
                return mod.register_forward_hook(fn, with_kwargs=True)
            except TypeError:
                return mod.register_forward_hook(fn)

        for spec in specs:
            module = spec.module
            module_id = id(module)

            clone_out = isinstance(module, self.auto_cache_clone_outputs_for_types)
            clone_in = bool(self.auto_cache_clone_inputs)

            def _pre_hook(*hook_args, _mid=module_id, _spec=spec, _clone_in=clone_in):
                _module, in_args, in_kwargs = self._parse_pre_hook_args(hook_args)

                if _spec.in_select is not None:
                    picked = _spec.in_select(in_args, in_kwargs)
                else:
                    picked = in_args[0] if len(in_args) > 0 else None

                x_in = self._pack_cache_tensor(picked, clone=_clone_in)
                pending_by_module[_mid].append(x_in)

            def _fwd_hook(*hook_args, _mid=module_id, _spec=spec, _clone_out=clone_out):
                _module, in_args, in_kwargs, output = self._parse_fwd_hook_args(hook_args)

                # Retrieve input captured by pre-hook (handles multiple calls)
                if pending_by_module[_mid]:
                    x_in = pending_by_module[_mid].pop(0)
                else:
                    # fallback: repack directly from current inputs
                    if _spec.in_select is not None:
                        picked_in = _spec.in_select(in_args, in_kwargs)
                    else:
                        picked_in = in_args[0] if len(in_args) > 0 else None
                    x_in = self._pack_cache_tensor(picked_in, clone=self.auto_cache_clone_inputs)

                if _spec.out_select is not None:
                    picked_out = _spec.out_select(output)
                else:
                    picked_out = output

                u = self._pack_cache_tensor(picked_out, clone=_clone_out)

                if x_in is not None:
                    block_inputs[_spec.name] = x_in
                    block_inputs_all[_spec.name].append(x_in)
                if u is not None:
                    block_outputs[_spec.name] = u
                    block_outputs_all[_spec.name].append(u)

                if self.auto_cache_include_steps:
                    call_idx = call_count_by_name[_spec.name]
                    call_count_by_name[_spec.name] += 1
                    steps.append(
                        {
                            "name": _spec.name,
                            "call_idx": call_idx,
                            "type": _module.__class__.__name__,
                            "rep": _spec.rep,  # metadata only
                            "group": _spec.group,
                            "is_output": bool(_spec.is_output),  # can be corrected after forward
                            "x_in": x_in,
                            "u": u,
                        }
                    )

            hooks.append(_register_pre_hook(module, _pre_hook))
            hooks.append(_register_fwd_hook(module, _fwd_hook))

        try:
            out = super().__call__(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        # Learn true output block name from execution trace (so BlockSpec.is_output can be correct next time)
        runtime_output_name: str | None = None
        if self.auto_output_strategy == "last_executed" and self.auto_cache_include_steps and steps:
            runtime_output_name = steps[-1]["name"]
            setattr(self, "_lemodule_runtime_output_name", runtime_output_name)
            # Also correct is_output flags in steps
            for i in range(len(steps)):
                steps[i]["is_output"] = i == len(steps) - 1

        # Provide runtime specs with corrected output marking (useful for rules that read specs)
        # This avoids relying on model.blocks having been updated already.
        if runtime_output_name is not None:
            runtime_specs = []
            for s in specs:
                runtime_specs.append(
                    BlockSpec(
                        name=s.name,
                        module=s.module,
                        rep=s.rep,
                        is_output=s.name == runtime_output_name,
                        group=s.group,
                        params=s.params,
                        in_select=s.in_select,
                        out_select=s.out_select,
                    )
                )
        else:
            runtime_specs = specs

        cache: dict[str, Any] = {
            "block_inputs": block_inputs,
            "block_outputs": block_outputs,
            "block_inputs_all": dict(block_inputs_all),
            "block_outputs_all": dict(block_outputs_all),
            "block_specs_runtime": runtime_specs,
        }
        if self.auto_cache_include_steps:
            cache["steps"] = steps
        return out, cache

    def __call__(self, *args, **kwargs):
        # If subclass already supports return_cache explicitly, respect it.
        if self._forward_accepts_return_cache():
            return super().__call__(*args, **kwargs)

        if "return_cache" not in kwargs:
            return super().__call__(*args, **kwargs)

        want_cache = bool(kwargs.pop("return_cache"))
        if not want_cache:
            return super().__call__(*args, **kwargs)

        return self._auto_forward_with_cache(*args, **kwargs)
