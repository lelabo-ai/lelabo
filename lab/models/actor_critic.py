# lab/models/actor_critic.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp_utils import forward_mlp_with_cache


class ActorCriticDiscrete(nn.Module):
    """
    Actor-Critic pour actions discrètes (CartPole-like).
    forward(obs) -> {"logits": [B,A], "value": [B]}
    Optionnel: return_cache=True -> (out_dict, cache_dict)
      cache_dict = {"actor": cacheA, "critic": cacheV}
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        obs_dim = int(obs_dim)
        n_actions = int(n_actions)
        hidden = int(hidden)
        layers = int(layers)
        assert layers >= 1

        dims_actor = [obs_dim] + [hidden] * layers + [n_actions]
        dims_critic = [obs_dim] + [hidden] * layers + [1]

        self.actor_linears = nn.ModuleList([nn.Linear(dims_actor[i], dims_actor[i + 1]) for i in range(len(dims_actor) - 1)])
        self.critic_linears = nn.ModuleList([nn.Linear(dims_critic[i], dims_critic[i + 1]) for i in range(len(dims_critic) - 1)])

        # heads explicites
        self.actor_head = self.actor_linears[-1]
        self.critic_head = self.critic_linears[-1]

        # compat si ailleurs tu utilises model.actor / model.critic
        self.actor = nn.Sequential(*[m for pair in zip(self.actor_linears[:-1], [nn.Tanh()] * (len(self.actor_linears) - 1)) for m in pair] + [self.actor_head])
        self.critic = nn.Sequential(*[m for pair in zip(self.critic_linears[:-1], [nn.Tanh()] * (len(self.critic_linears) - 1)) for m in pair] + [self.critic_head])

    def forward(self, obs: torch.Tensor, return_cache: bool = False):
        def tanh(a): return torch.tanh(a)

        logits, cacheA = forward_mlp_with_cache(self.actor_linears, obs, tanh, return_cache=return_cache)
        value, cacheV = forward_mlp_with_cache(self.critic_linears, obs, tanh, return_cache=return_cache)
        value = value.squeeze(-1)

        out = {"logits": logits, "value": value}
        if return_cache:
            return out, {"actor": cacheA, "critic": cacheV}
        return out
