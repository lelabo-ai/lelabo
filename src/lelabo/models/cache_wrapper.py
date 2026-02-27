from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

from .blocks import LeModule, BlockSpec


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


class ModelCacheWrapper(LeModule):
    """Wrap any nn.Module and expose LeModule-compatible cache collection."""

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

    def forward(self, *args, return_cache: bool = False, **kwargs):
        if not return_cache:
            return self.model(*args, **kwargs)

        block_inputs: dict[str, torch.Tensor] = {}
        block_outputs: dict[str, torch.Tensor] = {}
        steps: list[dict[str, Any]] = []
        pending_by_module: dict[int, list[torch.Tensor | None]] = defaultdict(list)
        hooks = []

        for spec in self._block_specs:
            module = spec.module
            module_id = id(module)

            def _pre_hook(_module, inputs, _mid=module_id):
                x_in = self._pack_tensor(inputs[0] if inputs else None)
                pending_by_module[_mid].append(x_in)

            def _fwd_hook(_module, inputs, output, _mid=module_id, _spec=spec):
                if pending_by_module[_mid]:
                    x_in = pending_by_module[_mid].pop(0)
                else:
                    x_in = self._pack_tensor(inputs[0] if inputs else None)
                u = self._pack_tensor(output)

                if x_in is not None:
                    block_inputs[_spec.name] = x_in
                if u is not None:
                    block_outputs[_spec.name] = u

                if self.include_steps:
                    steps.append(
                        {
                            "name": _spec.name,
                            "type": _module.__class__.__name__,
                            "rep": _spec.rep,
                            "group": _spec.group,
                            "is_output": bool(_spec.is_output),
                            "x_in": x_in,
                            "u": u,
                        }
                    )

            hooks.append(module.register_forward_pre_hook(_pre_hook))
            hooks.append(module.register_forward_hook(_fwd_hook))

        try:
            out = self.model(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()

        cache: dict[str, Any] = {
            "block_inputs": block_inputs,
            "block_outputs": block_outputs,
        }
        if self.include_steps:
            cache["steps"] = steps

        return out, cache
