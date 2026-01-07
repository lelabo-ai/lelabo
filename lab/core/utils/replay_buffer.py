# core/replay_buffer.py
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class ReplayBatch:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    sp: torch.Tensor
    done: torch.Tensor

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.sp = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, s, a, r, sp, done) -> None:
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.sp[i] = sp
        self.done[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str) -> ReplayBatch:
        idx = np.random.randint(0, self.size, size=batch_size)
        s = torch.from_numpy(self.s[idx]).to(device)
        a = torch.from_numpy(self.a[idx]).to(device)
        r = torch.from_numpy(self.r[idx]).to(device)
        sp = torch.from_numpy(self.sp[idx]).to(device)
        done = torch.from_numpy(self.done[idx]).to(device)
        return ReplayBatch(s=s, a=a, r=r, sp=sp, done=done)
