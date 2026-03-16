"""Soft Actor-Critic algorithm implementation."""

# lab/rl/algorithms/sac.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...models.builtins.actor_critic import SquashedGaussianActor, DoubleQCritic
from ..replay_buffer import ReplayBuffer


class SACCriticTask:
    """Loss for DoubleQCritic."""

    def __init__(self):
        self._last: Dict[str, float] = {}

    def loss(self, model_out: Dict[str, torch.Tensor], y: Dict[str, Any]) -> torch.Tensor:
        q1 = model_out["q1"].view(-1)
        q2 = model_out["q2"].view(-1)
        target = y["target"].float().view(-1).to(q1.dtype)

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        loss = q1_loss + q2_loss

        with torch.no_grad():
            self._last = {
                "critic_loss": float(loss.item()),
                "q1_loss": float(q1_loss.item()),
                "q2_loss": float(q2_loss.item()),
                "q1_mean": float(q1.mean().item()),
                "q2_mean": float(q2.mean().item()),
                "target_mean": float(target.mean().item()),
            }
        return loss

    def metrics(self, model_out: Dict[str, torch.Tensor], y: Dict[str, Any]) -> Dict[str, float]:
        _ = (model_out, y)
        return dict(self._last)


class SACActorTask:
    """Loss actor SAC (reparam + tanh squash) with an external DoubleQCritic."""

    def __init__(
        self,
        critic,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        alpha: float = 0.2,
    ):
        self.critic = critic
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.alpha = float(alpha)

        self._last: Dict[str, float] = {}
        self.last_log_pi_mean: Optional[torch.Tensor] = None

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def loss(self, model_out: Dict[str, torch.Tensor], y: Dict[str, Any]) -> torch.Tensor:
        obs = y["obs"]
        mu = model_out["mu"]
        log_std = model_out["log_std"]

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        std = log_std.exp()

        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        log_prob = -0.5 * (
            eps.pow(2) + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi, device=mu.device))
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_det = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        log_pi = log_prob - log_det

        q = self.critic(obs, action)
        min_q = torch.min(q["q1"], q["q2"]).view(-1, 1)
        loss = (self.alpha * log_pi - min_q).mean()

        with torch.no_grad():
            self.last_log_pi_mean = log_pi.mean().detach()
            self._last = {
                "actor_loss": float(loss.item()),
                "entropy": float((-log_pi).mean().item()),
                "log_pi": float(log_pi.mean().item()),
                "q_pi": float(min_q.mean().item()),
                "alpha": float(self.alpha),
            }
        return loss

    def metrics(self, model_out: Dict[str, torch.Tensor], y: Dict[str, Any]) -> Dict[str, float]:
        _ = (model_out, y)
        return dict(self._last)


@dataclass
class SACAlgoConfig:
    train_freq: int = 1
    learning_starts: int = 5_000

    gamma: float = 0.99
    tau: float = 0.005

    buffer_size: int = int(1e6)
    batch_size: int = 256
    gradient_steps: int = 1
    policy_frequency: int = 2
    target_network_frequency: int = 1

    alpha: float = 0.2
    autotune: bool = True
    target_entropy: Optional[float] = None


class SAC:
    """Soft Actor-Critic (continuous actions), compatible RLRunner."""

    def __init__(
        self,
        actor: SquashedGaussianActor,
        critic: DoubleQCritic,
        actor_learner,
        critic_learner,
        cfg: SACAlgoConfig,
        *,
        alpha_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg

        self.critic_target = DoubleQCritic(
            obs_dim=critic.obs_dim,
            act_dim=critic.act_dim,
            hidden_dim=critic.hidden_dim,
            num_layers=critic.num_layers,
            activation=critic.activation,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_learner = actor_learner
        self.critic_learner = critic_learner

        self.replay = ReplayBuffer(
            capacity=cfg.buffer_size,
            obs_dim=actor.obs_dim,
            action_shape=(actor.act_dim,),
            action_dtype=np.float32,
        )

        # entropy coefficient
        self.alpha = float(cfg.alpha)
        self.log_alpha: Optional[torch.nn.Parameter] = None
        self.alpha_optimizer = alpha_optimizer

        if cfg.autotune:
            init = float(cfg.alpha)
            if init <= 0:
                raise ValueError("SAC: alpha doit être > 0")
            self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor([init], dtype=torch.float32)))
            self.alpha = float(self.log_alpha.exp().item())
            if self.alpha_optimizer is None:
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.target_entropy = -float(actor.act_dim) if cfg.target_entropy is None else float(cfg.target_entropy)

        self.critic_task = SACCriticTask()
        self.actor_task = SACActorTask(
            critic=self.critic,
            action_scale=self.actor.action_scale,
            action_bias=self.actor.action_bias,
            alpha=self.alpha,
        )

        self.total_steps = 0
        self._obs: Optional[np.ndarray] = None
        self._ep_ret = 0.0
        self._ep_len = 0
        self._updates = 0

    def to(self, device: str):
        self.actor.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        if self.log_alpha is not None:
            self.log_alpha.data = self.log_alpha.data.to(device)
            if self.log_alpha.grad is not None:
                self.log_alpha.grad.data = self.log_alpha.grad.data.to(device)
        self.actor_task.action_scale = self.actor.action_scale
        self.actor_task.action_bias = self.actor.action_bias
        return self

    @torch.no_grad()
    def act(self, obs_t: torch.Tensor) -> np.ndarray:
        a, _, _ = self.actor.sample(obs_t.unsqueeze(0))
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act_deterministic(self, obs_t: torch.Tensor) -> np.ndarray:
        a = self.actor.act_deterministic(obs_t.unsqueeze(0))
        return a.squeeze(0).cpu().numpy()

    def collect(self, env, device: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._obs is None:
            obs, _ = env.reset()
            self._obs = np.asarray(obs, dtype=np.float32)
            self._ep_ret = 0.0
            self._ep_len = 0

        episodes = []

        for _ in range(int(self.cfg.train_freq)):
            assert self._obs is not None

            if int(self.total_steps) < int(self.cfg.learning_starts):
                a = env.action_space.sample()
            else:
                obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=device)
                a = self.act(obs_t)

            a = np.asarray(a, dtype=np.float32)

            sp, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            sp = np.asarray(sp, dtype=np.float32)

            self.replay.add(self._obs, a, float(r), sp, done)
            self.total_steps += 1

            self._ep_ret += float(r)
            self._ep_len += 1
            self._obs = sp

            if done:
                episodes.append({"return": float(self._ep_ret), "length": int(self._ep_len)})
                obs, _ = env.reset()
                self._obs = np.asarray(obs, dtype=np.float32)
                self._ep_ret = 0.0
                self._ep_len = 0

        return {"t": "sac_collect"}, {"episodes": episodes}

    def update(self, _batch: Any = None, device: str = "cpu") -> Dict[str, float]:
        if len(self.replay) < int(self.cfg.batch_size):
            return {}
        if int(self.total_steps) < int(self.cfg.learning_starts):
            return {}

        self._updates += 1
        stats_acc: Dict[str, float] = {}
        n_stats = 0

        for _ in range(int(self.cfg.gradient_steps)):
            data = self.replay.sample(int(self.cfg.batch_size), device=device)
            s = data.s
            a = data.a
            r = data.r.view(-1)
            sp = data.sp
            done = data.done.view(-1)

            with torch.no_grad():
                ap, log_pi_p, _ = self.actor.sample(sp)
                q_t = self.critic_target(sp, ap)
                min_q_t = torch.min(q_t["q1"], q_t["q2"])
                target = r + (1.0 - done) * float(self.cfg.gamma) * (min_q_t - self.alpha * log_pi_p.view(-1))

            xq = torch.cat([s, a], dim=-1)
            yq = {"target": target}
            critic_stats = self.critic_learner.train_step(self.critic, self.critic_task, (xq, yq), device)

            actor_stats: Dict[str, float] = {}
            alpha_stats: Dict[str, float] = {}

            if (self._updates % int(self.cfg.policy_frequency)) == 0:
                for p in self.critic.parameters():
                    p.requires_grad_(False)

                self.actor_task.set_alpha(self.alpha)
                actor_stats = self.actor_learner.train_step(self.actor, self.actor_task, (s, {"obs": s}), device)

                for p in self.critic.parameters():
                    p.requires_grad_(True)

                if self.cfg.autotune and self.log_alpha is not None and self.alpha_optimizer is not None:
                    log_pi_mean = self.actor_task.last_log_pi_mean
                    if log_pi_mean is not None:
                        alpha_loss = -(self.log_alpha.exp() * (log_pi_mean + self.target_entropy))
                        self.alpha_optimizer.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        self.alpha_optimizer.step()
                        self.alpha = float(self.log_alpha.exp().item())
                        alpha_stats = {"alpha_loss": float(alpha_loss.item()), "alpha": float(self.alpha)}

            if (self._updates % int(self.cfg.target_network_frequency)) == 0:
                self._soft_update(self.critic, self.critic_target, tau=float(self.cfg.tau))

            merged = {}
            merged.update({k: float(v) for k, v in (critic_stats or {}).items() if isinstance(v, (int, float))})
            merged.update({f"actor.{k}": float(v) for k, v in (actor_stats or {}).items() if isinstance(v, (int, float))})
            merged.update({k: float(v) for k, v in (alpha_stats or {}).items() if isinstance(v, (int, float))})

            for k, v in merged.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + float(v)
            n_stats += 1

        if n_stats > 0:
            for k in list(stats_acc.keys()):
                stats_acc[k] /= float(n_stats)

        stats_acc.setdefault("alpha", float(self.alpha))
        return stats_acc

    @staticmethod
    @torch.no_grad()
    def _soft_update(src: torch.nn.Module, tgt: torch.nn.Module, tau: float) -> None:
        tau = float(tau)
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    @torch.no_grad()
    def evaluate(self, env, eval_episodes: int, device: str) -> Dict[str, Any]:
        rets = []
        lens = []
        for _ in range(int(eval_episodes)):
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                a = self.act_deterministic(obs_t)
                obs, r, terminated, truncated, _ = env.step(np.asarray(a, dtype=np.float32))
                done = bool(terminated or truncated)
                obs = np.asarray(obs, dtype=np.float32)
                ep_ret += float(r)
                ep_len += 1
            rets.append(ep_ret)
            lens.append(ep_len)

        return {
            "mean_return": float(np.mean(rets)) if rets else 0.0,
            "mean_length": float(np.mean(lens)) if lens else 0.0,
            "n": int(len(rets)),
        }
"""Soft Actor-Critic algorithm implementation."""
