# algorithms/rl/dqn.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import copy
import torch

from core.utils.replay_buffer import ReplayBuffer
from core.task import DQNTask

@dataclass
class DQNConfig:
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 100_000
    learning_starts: int = 1000
    train_freq: int = 1
    target_update_freq: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

class DQN:
    """
    DQN orchestrates RL (collect + replay + targets),
    BUT delegates optimization to `learner.train_step(model, task, batch, device)`.
    """
    def __init__(self, q_net: torch.nn.Module, learner, cfg: DQNConfig):
        self.q = q_net
        self.q_target = copy.deepcopy(q_net)
        self.learner = learner
        self.cfg = cfg
        self.task = DQNTask()

        # infer dims
        self.n_actions: Optional[int] = None

        self.buffer: Optional[ReplayBuffer] = None
        self.total_steps = 0
        self.num_updates = 0

    def setup(self, obs_dim: int, n_actions: int):
        self.n_actions = int(n_actions)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim=obs_dim)

    def to(self, device: str):
        self.q.to(device)
        self.q_target.to(device)

    def epsilon(self) -> float:
        t = self.total_steps
        if t >= self.cfg.eps_decay_steps:
            return self.cfg.eps_end
        frac = t / max(1, self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> int:
        eps = self.epsilon()
        self.total_steps += 1

        if torch.rand(()) < eps:
            return int(torch.randint(0, self.n_actions, ()).item())

        qvals = self.q(obs.unsqueeze(0))
        return int(torch.argmax(qvals, dim=-1).item())

    def observe(self, s, a, r, sp, done) -> None:
        assert self.buffer is not None
        self.buffer.add(s, a, r, sp, done)

    def maybe_update(self, device: str) -> Dict[str, float]:
        assert self.buffer is not None

        if len(self.buffer) < self.cfg.learning_starts:
            return {}

        if (self.total_steps % self.cfg.train_freq) != 0:
            return {}

        batch = self.buffer.sample(self.cfg.batch_size, device=device)

        with torch.no_grad():
            q_next = self.q_target(batch.sp).max(dim=1).values
            target = batch.r + self.cfg.gamma * (1.0 - batch.done) * q_next

        # We pass (x, y_dict) so Backprop can handle y as Mapping
        x = batch.s
        y = {"action": batch.a, "target": target}

        stats = self.learner.train_step(self.q, self.task, (x, y), device)
        self.num_updates += 1

        if (self.num_updates % self.cfg.target_update_freq) == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        stats = dict(stats)
        stats["epsilon"] = float(self.epsilon())
        stats["buffer_size"] = float(len(self.buffer))
        stats["updates"] = float(self.num_updates)
        return stats
