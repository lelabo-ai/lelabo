from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

from .blocks import BlockSpec, normalize_standard_cache


def _first_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for v in obj:
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if hasattr(obj, "to_tuple"):
        try:
            return _first_tensor(obj.to_tuple())
        except Exception:
            return None
    return None


class ModelCacheWrapper(nn.Module):
    """Wrap any nn.Module and expose standard framework cache collection."""

    def __init__(
        self,
        model: nn.Module,
        *,
        block_specs: Sequence[BlockSpec] | None = None,
        block_filter: Callable[[str, nn.Module], bool] | None = None,
        detach_cache: bool = True,
        cache_to_cpu: bool = False,
        include_steps: bool = True,
    ):
        super().__init__()
        self.model = model
        self.detach_cache = bool(detach_cache)
        self.cache_to_cpu = bool(cache_to_cpu)
        self.include_steps = bool(include_steps)

        self._block_specs = self._build_block_specs(block_specs, block_filter)
        self._validate_specs(self._block_specs)

    def _build_block_specs(
        self,
        block_specs: Sequence[BlockSpec] | None,
        block_filter: Callable[[str, nn.Module], bool] | None,
    ) -> list[BlockSpec]:
        if block_specs is not None:
            specs = list(block_specs)
        else:
            pred = block_filter or (lambda _n, m: isinstance(m, (nn.Linear, nn.Conv2d)))
            specs = [
                BlockSpec(name=name, module=mod, rep="identity", is_output=False)
                for name, mod in self.model.named_modules()
                if name and pred(name, mod)
            ]

        if not specs:
            raise ValueError("ModelCacheWrapper: no selected blocks.")

        if not any(s.is_output for s in specs):
            last = specs[-1]
            specs[-1] = BlockSpec(
                name=last.name,
                module=last.module,
                rep=last.rep,
                is_output=True,
                group=last.group,
                params=last.params,
            )
        return specs

    @staticmethod
    def _validate_specs(specs: Sequence[BlockSpec]) -> None:
        seen: set[int] = set()
        for spec in specs:
            mid = id(spec.module)
            if mid in seen:
                raise ValueError(
                    "ModelCacheWrapper: each block must map to a distinct module instance."
                )
            seen.add(mid)

    def get_blocks(self) -> list[BlockSpec]:
        return list(self._block_specs)

    def _pack_tensor(self, value: Any) -> torch.Tensor | None:
        t = _first_tensor(value)
        if t is None:
            return None
        if self.detach_cache:
            t = t.detach()
        if self.cache_to_cpu:
            t = t.cpu()
        return t

    @staticmethod
    def _parse_pre_hook_args(hook_args: tuple[Any, ...]) -> tuple[nn.Module, tuple[Any, ...], dict[str, Any] | None]:
        if len(hook_args) == 2:
            module, args = hook_args
            return module, args, None
        if len(hook_args) == 3:
            module, args, kwargs = hook_args
            return module, args, kwargs
        raise TypeError(f"Unexpected forward_pre_hook signature: {len(hook_args)} args")

    @staticmethod
    def _parse_fwd_hook_args(hook_args: tuple[Any, ...]) -> tuple[nn.Module, tuple[Any, ...], dict[str, Any] | None, Any]:
        if len(hook_args) == 3:
            module, args, output = hook_args
            return module, args, None, output
        if len(hook_args) == 4:
            module, args, kwargs, output = hook_args
            return module, args, kwargs, output
        raise TypeError(f"Unexpected forward_hook signature: {len(hook_args)} args")

    def forward(self, *args, return_cache: bool = False, **kwargs):
        if not return_cache:
            return self.model(*args, **kwargs)

        block_inputs: dict[str, torch.Tensor] = {}
        block_outputs: dict[str, torch.Tensor] = {}
        block_inputs_all: dict[str, list[torch.Tensor]] = defaultdict(list)
        block_outputs_all: dict[str, list[torch.Tensor]] = defaultdict(list)
        steps: list[dict[str, Any]] = []
        pending_by_module: dict[int, list[torch.Tensor | None]] = defaultdict(list)
        call_count_by_name: dict[str, int] = defaultdict(int)
        hooks = []

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

        for spec in self._block_specs:
            module = spec.module
            module_id = id(module)

            def _pre_hook(*hook_args, _mid=module_id, _spec=spec):
                _module, in_args, in_kwargs = self._parse_pre_hook_args(hook_args)
                if _spec.in_select is not None:
                    picked = _spec.in_select(in_args, in_kwargs)
                else:
                    picked = in_args[0] if in_args else None
                x_in = self._pack_tensor(picked)
                pending_by_module[_mid].append(x_in)

            def _fwd_hook(*hook_args, _mid=module_id, _spec=spec):
                _module, in_args, in_kwargs, output = self._parse_fwd_hook_args(hook_args)
                if pending_by_module[_mid]:
                    x_in = pending_by_module[_mid].pop(0)
                else:
                    if _spec.in_select is not None:
                        picked_in = _spec.in_select(in_args, in_kwargs)
                    else:
                        picked_in = in_args[0] if in_args else None
                    x_in = self._pack_tensor(picked_in)

                picked_out = _spec.out_select(output) if _spec.out_select is not None else output
                out_t = self._pack_tensor(picked_out)

                if x_in is not None:
                    block_inputs[_spec.name] = x_in
                    block_inputs_all[_spec.name].append(x_in)
                if out_t is not None:
                    block_outputs[_spec.name] = out_t
                    block_outputs_all[_spec.name].append(out_t)

                if self.include_steps:
                    call_idx = call_count_by_name[_spec.name]
                    call_count_by_name[_spec.name] += 1
                    steps.append(
                        {
                            "name": _spec.name,
                            "call_idx": call_idx,
                            "type": _module.__class__.__name__,
                            "rep": _spec.rep,
                            "group": _spec.group,
                            "is_output": bool(_spec.is_output),
                            "x_in": x_in,
                            "out": out_t,
                        }
                    )

            hooks.append(_register_pre_hook(module, _pre_hook))
            hooks.append(_register_fwd_hook(module, _fwd_hook))

        try:
            out = self.model(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        cache: dict[str, Any] = {
            "cache_version": "standard.v1",
            "block_inputs": block_inputs,
            "block_outputs": block_outputs,
            "block_inputs_all": dict(block_inputs_all),
            "block_outputs_all": dict(block_outputs_all),
            "block_specs_runtime": list(self._block_specs),
        }
        if self.include_steps:
            cache["steps"] = steps

        return out, normalize_standard_cache(cache)
