# models/actor_critic.py
from __future__ import annotations
import torch
import torch.nn as nn


class ActorCriticDiscrete(nn.Module):
    """
    Actor-Critic pour actions discrètes (ex: CartPole).
    forward(obs) -> dict: {"logits": [B,A], "value": [B]}
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        assert layers >= 1

        def mlp_head(out_dim: int):
            mods = []
            in_dim = obs_dim
            for _ in range(layers):
                mods += [nn.Linear(in_dim, hidden), nn.Tanh()]
                in_dim = hidden
            mods += [nn.Linear(in_dim, out_dim)]
            return nn.Sequential(*mods)

        self.actor = mlp_head(n_actions)
        self.critic = mlp_head(1)

    def forward(self, obs: torch.Tensor) -> dict:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return {"logits": logits, "value": value}
