# lab/models/blocks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch.nn as nn


@dataclass(frozen=True)
class BlockSpec:
    """Describe a module block to enable modular/local update rules.

    - name: stable identifier (used in caches)
    - module: nn.Module that implements the block forward
    - rep: representation hint ("identity", "gap", "mean", "cls", ...)
    - is_output: True for the final prediction head
    - group: optional namespace (e.g. "actor" / "critic" for branches)
    - params: optional explicit parameter list (when module.parameters() isn't the
      right thing, or when you intentionally wrap without registering params).
    """

    name: str
    module: nn.Module
    rep: str = "identity"
    is_output: bool = False
    group: str = "main"
    params: Optional[Sequence[nn.Parameter]] = None

    def iter_params(self):
        if self.params is not None:
            yield from self.params
        else:
            yield from self.module.parameters()


class BlockModel(nn.Module):
    """Mixin: models expose blocks + return_cache with a shared convention."""

    def get_blocks(self) -> list[BlockSpec]:
        raise NotImplementedError

    @property
    def blocks(self) -> list[BlockSpec]:
        return list(self.get_blocks())

    # Backward-compat: some update rules may expect this older name.
    @property
    def local_blocks(self):
        return [
            {
                "name": b.name,
                "module": b.module,
                "rep": b.rep,
                "is_output": b.is_output,
                "group": b.group,
            }
            for b in self.blocks
        ]
