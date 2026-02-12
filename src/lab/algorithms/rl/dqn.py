# lab/algorithms/rl/dqn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import copy

import numpy as np
import torch

from ...core.replay_buffer import ReplayBuffer
from ...core.task import DQNTask


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
    DQN = algo + interaction env (collect) + update.
    Compatible avec un runner unique RLRunner.
    """
    def __init__(self, q_net: torch.nn.Module, learner, cfg: DQNConfig):
        self.q = q_net
        self.q_target = copy.deepcopy(q_net)
        self.learner = learner
        self.cfg = cfg
        self.task = DQNTask()

        self.n_actions: Optional[int] = None
        self.buffer: Optional[ReplayBuffer] = None

        self.total_steps = 0
        self.num_updates = 0

        # état env pour collect()
        self._obs = None
        self._ep_return = 0.0
        self._ep_len = 0
        self._episode_count = 0

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
    def _act(self, obs: torch.Tensor, explore: bool = True) -> int:
        assert self.n_actions is not None
        if explore:
            eps = self.epsilon()
            if torch.rand(()) < eps:
                return int(torch.randint(0, self.n_actions, ()).item())

        qvals = self.q(obs.unsqueeze(0))
        return int(torch.argmax(qvals, dim=-1).item())

    def collect(self, env, device: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fait 1 step d'env, stocke dans replay.
        Retourne (batch, info). Ici batch vide: update() tire dans le replay.
        """
        assert self.buffer is not None

        if self._obs is None:
            obs, _ = env.reset()
            self._obs = obs
            self._ep_return = 0.0
            self._ep_len = 0

        obs_t = torch.tensor(self._obs, dtype=torch.float32, device=device)
        action = self._act(obs_t, explore=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        self.buffer.add(self._obs, action, float(reward), next_obs, done)

        self.total_steps += 1
        self._ep_return += float(reward)
        self._ep_len += 1

        episodes: List[Dict[str, Any]] = []
        if done:
            self._episode_count += 1
            episodes.append(
                {
                    "episode": int(self._episode_count),
                    "step": int(self.total_steps),
                    "return": float(self._ep_return),
                    "length": int(self._ep_len),
                }
            )
            obs, _ = env.reset()
            self._obs = obs
            self._ep_return = 0.0
            self._ep_len = 0
        else:
            self._obs = next_obs

        return {}, {"episodes": episodes}

    def update(self, _batch: Dict[str, Any], device: str) -> Dict[str, float]:
        """
        Une update éventuelle (selon train_freq / learning_starts).
        """
        assert self.buffer is not None

        if len(self.buffer) < self.cfg.learning_starts:
            return {}

        if (self.total_steps % self.cfg.train_freq) != 0:
            return {}

        batch = self.buffer.sample(self.cfg.batch_size, device=device)

        with torch.no_grad():
            q_next = self.q_target(batch.sp).max(dim=1).values
            target = batch.r + self.cfg.gamma * (1.0 - batch.done) * q_next

        x = batch.s
        y = {"action": batch.a, "target": target}

        stats = self.learner.train_step(self.q, self.task, (x, y), device)
        self.num_updates += 1

        if (self.num_updates % self.cfg.target_update_freq) == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        out = dict(stats)
        out["epsilon"] = float(self.epsilon())
        out["buffer_size"] = float(len(self.buffer))
        out["updates"] = float(self.num_updates)
        return out

    @torch.no_grad()
    def evaluate(self, env, eval_episodes: int, device: str) -> Dict[str, Any]:
        returns = []
        lengths = []

        for _ in range(eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                action = self._act(obs_t, explore=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1

            returns.append(ep_ret)
            lengths.append(ep_len)

        mean_ret = float(np.mean(returns)) if returns else 0.0
        ci95 = 0.0
        if len(returns) > 1:
            m = mean_ret
            var = sum((x - m) ** 2 for x in returns) / (len(returns) - 1)
            se = (var ** 0.5) / (len(returns) ** 0.5)
            ci95 = float(1.96 * se)

        return {"mean_return": mean_ret, "ci95": float(ci95), "n": int(len(returns))}
