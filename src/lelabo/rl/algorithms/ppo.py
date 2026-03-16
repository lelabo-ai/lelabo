"""Proximal Policy Optimization algorithm implementation."""

# lab/rl/algorithms/ppo.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List

import numpy as np
import torch
from typing import Optional
from torch.distributions.categorical import Categorical

from .contract import list_contract_keys, resolve_dataclass_overrides


@dataclass
class PPOConfig:
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: Optional[float] = None


class PPOTask:
    """
    model_out: {"logits": [B,A], "value": [B]}
    y: dict containing:
      - "actions": LongTensor [B]
      - "old_logprobs": FloatTensor [B]
      - "advantages": FloatTensor [B]
      - "returns": FloatTensor [B]
      - "old_values": FloatTensor [B]
    """

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self._last: Dict[str, float] = {}

    def loss(self, model_out: Dict[str, torch.Tensor], y: Dict[str, Any]) -> torch.Tensor:
        logits = model_out["logits"]
        values = model_out["value"]

        actions = y["actions"].long()
        old_logprobs = y["old_logprobs"].float()
        advantages = y["advantages"].float()
        returns = y["returns"].float()
        old_values = y["old_values"].float()

        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        logratio = new_logprobs - old_logprobs
        ratio = logratio.exp()

        if self.cfg.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        if self.cfg.clip_vloss:
            v_unclipped = (values - returns) ** 2
            v_clipped = old_values + torch.clamp(values - old_values, -self.cfg.clip_coef, self.cfg.clip_coef)
            v_clipped_loss = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
        else:
            v_loss = 0.5 * ((values - returns) ** 2).mean()

        loss = pg_loss - self.cfg.ent_coef * entropy + self.cfg.vf_coef * v_loss

        with torch.no_grad():
            approx_kl = (ratio - 1.0 - logratio).mean().item()
            clipfrac = (torch.abs(ratio - 1.0) > self.cfg.clip_coef).float().mean().item()
            self._last = {
                "loss": float(loss.item()),
                "pg_loss": float(pg_loss.item()),
                "v_loss": float(v_loss.item()),
                "entropy": float(entropy.item()),
                "approx_kl": float(approx_kl),
                "clipfrac": float(clipfrac),
            }
        return loss

    def metrics(self, model_out: Dict[str, torch.Tensor], y: Dict[str, Any]) -> Dict[str, float]:
        _ = (model_out, y)
        return dict(self._last)

    @torch.no_grad()
    def output_deltas(self, model_out: Dict[str, torch.Tensor], y: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        logits = model_out["logits"]
        values = model_out["value"]
        values_ = values.view(-1, 1) if values.dim() == 1 else values

        actions = y["actions"].long()
        old_logprobs = y["old_logprobs"].float()
        advantages = y["advantages"].float()
        returns = y["returns"].float()
        old_values = y["old_values"].float()

        batch_size = logits.size(0)
        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(actions)
        logratio = new_logprobs - old_logprobs
        ratio = logratio.exp()

        if self.cfg.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        clip_coef = float(self.cfg.clip_coef)
        pg1 = -advantages * ratio
        r_clamped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg2 = -advantages * r_clamped

        use_pg1 = pg1 > pg2
        in_clip = (ratio >= (1.0 - clip_coef)) & (ratio <= (1.0 + clip_coef))
        d_pg1_dlogp = -advantages * ratio
        d_pg2_dlogp = torch.where(in_clip, -advantages * ratio, torch.zeros_like(ratio))
        d_loss_dlogp = torch.where(use_pg1, d_pg1_dlogp, d_pg2_dlogp) / float(batch_size)

        pi = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(pi)
        onehot.scatter_(1, actions.view(-1, 1), 1.0)
        delta_logits_pg = d_loss_dlogp.view(-1, 1) * (onehot - pi)

        if self.cfg.ent_coef != 0.0:
            logp = torch.log(pi + 1e-8)
            series = (pi * (logp + 1.0)).sum(dim=1, keepdim=True)
            d_entropy_dz = pi * (series - (logp + 1.0))
            delta_logits_ent = (-float(self.cfg.ent_coef) / float(batch_size)) * d_entropy_dz
        else:
            delta_logits_ent = torch.zeros_like(delta_logits_pg)
        delta_logits = delta_logits_pg + delta_logits_ent

        returns_ = returns.view(-1, 1).to(values_.dtype)
        old_values_ = old_values.view(-1, 1).to(values_.dtype)

        if self.cfg.clip_vloss:
            v_unclipped = (values_ - returns_) ** 2
            v_clipped = old_values_ + torch.clamp(values_ - old_values_, -clip_coef, clip_coef)
            v_clipped_loss = (v_clipped - returns_) ** 2
            use_unclipped = v_unclipped >= v_clipped_loss
            d_unclipped = values_ - returns_
            unclamped = (values_ - old_values_).abs() <= clip_coef
            d_clipped = (v_clipped - returns_) * unclamped.to(values_.dtype)
            d_value = torch.where(use_unclipped, d_unclipped, d_clipped)
        else:
            d_value = values_ - returns_

        delta_value = (float(self.cfg.vf_coef) / float(batch_size)) * d_value
        return {"logits": delta_logits, "value": delta_value}


@dataclass
class PPOAlgoConfig:
    num_envs: int = 8
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    ppo: PPOConfig = field(default_factory=PPOConfig)


_PPO_CONFIG_ALIASES = {
    "clip_coef": "ppo.clip_coef",
    "ent_coef": "ppo.ent_coef",
    "vf_coef": "ppo.vf_coef",
    "norm_adv": "ppo.norm_adv",
    "clip_vloss": "ppo.clip_vloss",
    "target_kl": "ppo.target_kl",
}

_PPO_ALLOW_NONE_FLOAT_PATHS = {"ppo.target_kl"}


def get_config_contract() -> tuple[str, ...]:
    """Return accepted override keys for PPO config resolution."""
    return list_contract_keys(PPOAlgoConfig(), aliases=_PPO_CONFIG_ALIASES)


def resolve_config_overrides(overrides: dict[str, str]) -> PPOAlgoConfig:
    """Build PPO config from string overrides validated against PPO contract."""
    return resolve_dataclass_overrides(
        PPOAlgoConfig(),
        overrides,
        aliases=_PPO_CONFIG_ALIASES,
        allow_none_float_paths=_PPO_ALLOW_NONE_FLOAT_PATHS,
        algo_label="PPO",
    )


class PPO:
    """
    Compatible RLRunner :
      - collect(envs) -> rollout dict + episodes
      - update(rollout) -> stats
      - evaluate(env) -> mean_return
    Discrete actions.
    """
    def __init__(self, actor_critic: torch.nn.Module, learner, cfg: PPOAlgoConfig):
        self.model = actor_critic
        self.learner = learner
        self.cfg = cfg
        self.task = PPOTask(cfg.ppo)
        self.global_step = 0

        self._next_obs = None
        self._next_done = None

        self._ep_returns = None
        self._ep_lengths = None

    @property
    def total_steps(self) -> int:
        return int(self.global_step)

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

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> int:
        out = self.model(obs.unsqueeze(0))
        logits = out["logits"][0]
        return int(torch.argmax(logits, dim=-1).item())

    def collect(self, envs, device: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        roll = self.rollout(envs, device=device)
        eps = roll.get("_episodes", [])
        return roll, {"episodes": eps}

    def rollout(self, envs, device: str) -> Dict[str, Any]:
        cfg = self.cfg
        num_envs, num_steps = cfg.num_envs, cfg.num_steps

        obs_shape = envs.single_observation_space.shape
        assert len(obs_shape) == 1, "PPO ici suppose observations vector (CartPole-like)."

        obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device, dtype=torch.float32)
        actions = torch.zeros((num_steps, num_envs), device=device, dtype=torch.long)
        logprobs = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32)
        rewards = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32)
        dones = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32)
        values = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32)

        if self._next_obs is None:
            o, _ = envs.reset()
            self._next_obs = torch.as_tensor(o, device=device, dtype=torch.float32)
            self._next_done = torch.zeros(num_envs, device=device, dtype=torch.float32)

        if (
            self._ep_returns is None
            or self._ep_lengths is None
            or self._ep_returns.numel() != num_envs
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
            done_np = np.logical_or(term, trunc)

            r_t = torch.as_tensor(r, device=device, dtype=torch.float32)
            rewards[t] = r_t

            self._ep_returns += r_t
            self._ep_lengths += 1

            done_t = torch.as_tensor(done_np, device=device, dtype=torch.bool)
            if done_t.any():
                idx = torch.nonzero(done_t, as_tuple=False).squeeze(-1)
                for i in idx.tolist():
                    episodes_finished.append(
                        {"return": float(self._ep_returns[i].item()), "length": int(self._ep_lengths[i].item())}
                    )
                self._ep_returns[idx] = 0.0
                self._ep_lengths[idx] = 0

            next_obs = torch.as_tensor(o2, device=device, dtype=torch.float32)
            next_done = done_t.float()

            self.global_step += num_envs

        with torch.no_grad():
            next_value = self.model(next_obs)["value"]

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
            "_episodes": episodes_finished,
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

    def update(self, roll: Dict[str, Any], device: str) -> Dict[str, float]:
        cfg = self.cfg
        ppo_cfg = cfg.ppo

        adv, ret = self.compute_gae(roll)

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
        b_adv = adv.reshape(batch_size)
        b_returns = ret.reshape(batch_size)
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
                a = self.act_deterministic(obs_t)
                obs, reward, terminated, truncated, _ = env.step(a)
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
"""Proximal Policy Optimization algorithm implementation."""
