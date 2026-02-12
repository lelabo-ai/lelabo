# lab/core/replay_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class TransitionBatch:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    sp: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.s = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.sp = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return int(self._size)

    def add(self, s, a: int, r: float, sp, done: bool) -> None:
        i = self._idx
        self.s[i] = np.asarray(s, dtype=np.float32).reshape(-1)[: self.obs_dim]
        self.sp[i] = np.asarray(sp, dtype=np.float32).reshape(-1)[: self.obs_dim]
        self.a[i] = int(a)
        self.r[i] = float(r)
        self.done[i] = float(bool(done))

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int, device: str = "cpu", seed: Optional[int] = None) -> TransitionBatch:
        bs = int(batch_size)
        if self._size <= 0:
            raise ValueError("ReplayBuffer is empty")

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self._size, size=bs, endpoint=False)

        s = torch.as_tensor(self.s[idx], device=device)
        sp = torch.as_tensor(self.sp[idx], device=device)
        a = torch.as_tensor(self.a[idx], device=device)
        r = torch.as_tensor(self.r[idx], device=device)
        done = torch.as_tensor(self.done[idx], device=device)

        return TransitionBatch(s=s, a=a, r=r, sp=sp, done=done)
