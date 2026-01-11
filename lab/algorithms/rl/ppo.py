# lab/algorithms/rl/ppo.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from core.task import PPOConfig, PPOTask


@dataclass
class PPOAlgoConfig:
    # rollout
    num_envs: int = 8
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # update
    update_epochs: int = 4
    num_minibatches: int = 4

    # PPO loss config
    ppo: PPOConfig = field(default_factory=PPOConfig)


class PPO:
    """
    PPO orchestration (rollout + GAE + minibatch updates),
    delegates optimization to learner.train_step(model, task, (obs, y), device).

    Discrete actions (CartPole-like).
    Model must output {"logits": [B,A], "value": [B]}.
    """
    def __init__(self, actor_critic: torch.nn.Module, learner, cfg: PPOAlgoConfig):
        self.model = actor_critic
        self.learner = learner
        self.cfg = cfg
        self.task = PPOTask(cfg.ppo)
        self.global_step = 0

        # rollout state
        self._next_obs = None
        self._next_done = None

        # episode trackers (vector env)
        self._ep_returns = None   # torch.Tensor [N]
        self._ep_lengths = None   # torch.Tensor [N]

    def to(self, device: str):
        self.model.to(device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(obs)
        dist = Categorical(logits=out["logits"])
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        values = out["value"]
        return actions, logprobs, values

    def rollout(self, envs, device: str) -> Dict[str, Any]:
        cfg = self.cfg
        num_envs, num_steps = cfg.num_envs, cfg.num_steps

        obs_shape = envs.single_observation_space.shape
        assert len(obs_shape) == 1, "This PPO implementation assumes vector observations (CartPole-like)."

        obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device, dtype=torch.float32)
        actions = torch.zeros((num_steps, num_envs), device=device, dtype=torch.long)
        logprobs = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32)
        rewards = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32)
        dones = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32)
        values = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32)

        # init rollout state
        if self._next_obs is None:
            o, _ = envs.reset()
            self._next_obs = torch.as_tensor(o, device=device, dtype=torch.float32)
            self._next_done = torch.zeros(num_envs, device=device, dtype=torch.float32)

        # init episode trackers (robuste, sans dépendre de info["episode"])
        if (
            self._ep_returns is None
            or self._ep_lengths is None
            or self._ep_returns.numel() != num_envs
            or self._ep_returns.device.type != torch.device(device).type
        ):
            self._ep_returns = torch.zeros(num_envs, device=device, dtype=torch.float32)
            self._ep_lengths = torch.zeros(num_envs, device=device, dtype=torch.int64)

        next_obs = self._next_obs
        next_done = self._next_done

        episodes_finished: List[Dict[str, float]] = []

        for t in range(num_steps):
            obs[t] = next_obs
            dones[t] = next_done

            with torch.no_grad():
                a, lp, v = self.act(next_obs)

            actions[t] = a
            logprobs[t] = lp
            values[t] = v

            a_np = a.detach().cpu().numpy()
            o2, r, term, trunc, _info = envs.step(a_np)

            done_np = np.logical_or(term, trunc)  # np.bool_ array shape [N]

            r_t = torch.as_tensor(r, device=device, dtype=torch.float32)
            rewards[t] = r_t

            # ---- EPISODE TRACKING (robust) ----
            self._ep_returns += r_t
            self._ep_lengths += 1

            done_t = torch.as_tensor(done_np, device=device, dtype=torch.bool)
            if done_t.any():
                idx = torch.nonzero(done_t, as_tuple=False).squeeze(-1)
                for i in idx.tolist():
                    episodes_finished.append(
                        {"r": float(self._ep_returns[i].item()), "l": float(self._ep_lengths[i].item())}
                    )
                self._ep_returns[idx] = 0.0
                self._ep_lengths[idx] = 0

            # next state
            next_obs = torch.as_tensor(o2, device=device, dtype=torch.float32)
            next_done = done_t.float()

            self.global_step += num_envs

        with torch.no_grad():
            next_value = self.model(next_obs)["value"]  # [N]

        self._next_obs = next_obs
        self._next_done = next_done

        return {
            "obs": obs,
            "actions": actions,
            "logprobs": logprobs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "next_value": next_value,
            "_episodes": episodes_finished,  # list of {r,l} for this rollout
        }

    @torch.no_grad()
    def compute_gae(self, roll: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        rewards = roll["rewards"]
        dones = roll["dones"]
        values = roll["values"]
        next_value = roll["next_value"]

        T, N = rewards.shape
        advantages = torch.zeros((T, N), device=rewards.device, dtype=torch.float32)

        lastgaelam = torch.zeros((N,), device=rewards.device, dtype=torch.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        roll: Dict[str, Any],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        device: str,
    ) -> Dict[str, float]:
        cfg = self.cfg
        ppo_cfg = cfg.ppo

        obs = roll["obs"]
        actions = roll["actions"]
        old_logprobs = roll["logprobs"]
        old_values = roll["values"]

        T, N = actions.shape
        batch_size = T * N
        minibatch_size = max(1, batch_size // cfg.num_minibatches)

        b_obs = obs.reshape((batch_size,) + obs.shape[2:])
        b_actions = actions.reshape(batch_size)
        b_old_logprobs = old_logprobs.reshape(batch_size)
        b_adv = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_old_values = old_values.reshape(batch_size)

        agg: Dict[str, float] = {}
        n_stats = 0

        for _epoch in range(cfg.update_epochs):
            inds = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start : start + minibatch_size]

                x = b_obs[mb]
                y = {
                    "actions": b_actions[mb],
                    "old_logprobs": b_old_logprobs[mb],
                    "advantages": b_adv[mb],
                    "returns": b_returns[mb],
                    "old_values": b_old_values[mb],
                }

                stats = self.learner.train_step(self.model, self.task, (x, y), device)

                if ppo_cfg.target_kl is not None:
                    akl = float(stats.get("approx_kl", 0.0))
                    if akl > ppo_cfg.target_kl:
                        break

                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        agg[k] = agg.get(k, 0.0) + float(v)
                n_stats += 1

        if n_stats > 0:
            for k in list(agg.keys()):
                agg[k] /= n_stats

        return agg
