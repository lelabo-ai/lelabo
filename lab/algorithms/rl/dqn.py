# algorithms/rl/dqn.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from core.utils.replay_buffer import ReplayBuffer
from algorithms.update_rules.base import UpdateRule


@dataclass
class DQNConfig:
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 100_000
    learning_starts: int = 1000
    train_freq: int = 1            # update every N env steps
    target_update_freq: int = 1000 # hard update every N updates
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000  # linear decay steps
    max_grad_norm: Optional[float] = None


class DQN:
    """
    Algorithme RL (DQN) qui s'appuie sur une UpdateRule pour optimiser le Q-network.
    """
    def __init__(self, q_net: torch.nn.Module, update_rule: UpdateRule, obs_dim: int, n_actions: int, cfg: DQNConfig):
        self.q = q_net
        self.q_target = type(q_net)(*[])  # will be replaced by deepcopy below (safe fallback)
        self.q_target = torch.deepcopy(q_net) if hasattr(torch, "deepcopy") else None  # torch>=2.1 maybe
        if self.q_target is None:
            import copy
            self.q_target = copy.deepcopy(q_net)

        self.update_rule = update_rule
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg

        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim=obs_dim)
        self.total_steps = 0
        self.num_updates = 0

    def to(self, device: str):
        self.q.to(device)
        self.q_target.to(device)

    def epsilon(self) -> float:
        # linear decay
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

        qvals = self.q(obs.unsqueeze(0))  # [1, A]
        return int(torch.argmax(qvals, dim=-1).item())

    def observe(self, s, a, r, sp, done) -> None:
        self.buffer.add(s, a, r, sp, done)

    def maybe_update(self, device: str) -> Dict[str, float]:
        """
        Réalise éventuellement un update (selon train_freq / learning_starts).
        Retourne des stats (loss, etc.) ou dict vide.
        """
        if len(self.buffer) < self.cfg.learning_starts:
            return {}

        # update every train_freq steps (based on env interaction steps)
        if (self.total_steps % self.cfg.train_freq) != 0:
            return {}

        batch = self.buffer.sample(self.cfg.batch_size, device=device)

        # Q(s,a)
        q_sa = self.q(batch.s).gather(1, batch.a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            # Double DQN could be added later; here plain DQN target
            q_next = self.q_target(batch.sp).max(dim=1).values
            target = batch.r + self.cfg.gamma * (1.0 - batch.done) * q_next

        loss = F.smooth_l1_loss(q_sa, target)

        out = self.update_rule.step(self.q, loss)
        self.num_updates += 1

        # hard update target
        if (self.num_updates % self.cfg.target_update_freq) == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        out.update({
            "td_loss": float(loss.detach().item()),
            "epsilon": float(self.epsilon()),
            "buffer_size": float(len(self.buffer)),
            "updates": float(self.num_updates),
        })
        return out
