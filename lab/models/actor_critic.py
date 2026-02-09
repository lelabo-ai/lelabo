# lab/models/actor_critic.py
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import BlockModel, BlockSpec
from .mlp import MLPStack


class ActorCriticDiscrete(BlockModel):
    """
    Standard Actor-Critic for discrete actions, built from TWO standard MLPStack.

    - actor: outputs logits [B, n_actions]
    - critic: outputs value [B] (from a Linear head of size 1)

    This keeps everything compatible with:
      - FA/DFA/TargetProp: needs cache["inputs"]/["preacts"] (provided by each MLPStack)
      - SoftHebb/KP and any block-based rule: needs cache["block_inputs"] and model.get_blocks()
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = str(activation).lower()

        actor_dims = [self.obs_dim] + [self.hidden_dim] * self.num_layers + [self.n_actions]
        critic_dims = [self.obs_dim] + [self.hidden_dim] * self.num_layers + [1]

        self.actor = MLPStack(actor_dims, activation=self.activation)
        self.critic = MLPStack(critic_dims, activation=self.activation)

        # handy aliases (some algos / logs might like these)
        self.actor_linears = self.actor.linears
        self.critic_linears = self.critic.linears
        self.actor_head: nn.Linear = self.actor.head
        self.critic_head: nn.Linear = self.critic.head

    def get_blocks(self) -> list[BlockSpec]:
        blocks: list[BlockSpec] = []

        # Actor blocks (names MUST match cache["block_inputs"] keys below)
        for i, lin in enumerate(self.actor_linears):
            is_out = (i == (len(self.actor_linears) - 1))
            name = f"actor.layer{i}" if not is_out else "actor.head"
            blocks.append(BlockSpec(name=name, module=lin, rep="identity", is_output=is_out, group="actor"))

        # Critic blocks
        for i, lin in enumerate(self.critic_linears):
            is_out = (i == (len(self.critic_linears) - 1))
            name = f"critic.layer{i}" if not is_out else "critic.head"
            blocks.append(BlockSpec(name=name, module=lin, rep="identity", is_output=is_out, group="critic"))

        return blocks

    def forward(self, obs: torch.Tensor, return_cache: bool = False):
        if not return_cache:
            logits = self.actor(obs)
            value = self.critic(obs).squeeze(-1)  # [B]
            return {"logits": logits, "value": value}

        logits, cacheA = self.actor(obs, return_cache=True)
        value, cacheV = self.critic(obs, return_cache=True)
        value = value.squeeze(-1)

        # Merge block_inputs with stable prefixed keys
        merged_block_inputs = {}

        # actor cacheA["block_inputs"] has keys: "layer{i}", "head"
        for k, v in cacheA.get("block_inputs", {}).items():
            merged_block_inputs[f"actor.{k}"] = v

        # critic cacheV["block_inputs"] has keys: "layer{i}", "head"
        for k, v in cacheV.get("block_inputs", {}).items():
            merged_block_inputs[f"critic.{k}"] = v

        cache = {
            "actor": cacheA,
            "critic": cacheV,
            "block_inputs": merged_block_inputs,
        }
        return {"logits": logits, "value": value}, cache


# ============================================================
# SAC / continuous control (Mujoco, etc.)
# ============================================================

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


class SquashedGaussianActor(BlockModel):
    """Actor SAC: Gaussian squashed par tanh + rescaling vers Box."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low,
        action_high,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = str(activation).lower()

        dims = [self.obs_dim] + [self.hidden_dim] * self.num_layers + [2 * self.act_dim]
        self.net = MLPStack(dims, activation=self.activation)
        self.linears = self.net.linears

        low = torch.as_tensor(action_low, dtype=torch.float32).view(-1)
        high = torch.as_tensor(action_high, dtype=torch.float32).view(-1)
        if low.numel() != self.act_dim or high.numel() != self.act_dim:
            raise ValueError(f"action_low/high must have act_dim={self.act_dim} elements")

        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    @property
    def head(self) -> nn.Linear:
        return self.net.head

    def get_blocks(self) -> list[BlockSpec]:
        blocks: list[BlockSpec] = []
        for i, lin in enumerate(self.linears):
            is_out = (i == (len(self.linears) - 1))
            name = f"actor.layer{i}" if not is_out else "actor.head"
            blocks.append(BlockSpec(name=name, module=lin, rep="identity", is_output=is_out, group="actor"))
        return blocks

    def forward(self, obs: torch.Tensor, return_cache: bool = False):
        if not return_cache:
            raw = self.net(obs)
            mu, log_std = raw.chunk(2, dim=-1)
            return {"mu": mu, "log_std": log_std, "raw": raw}

        raw, cache = self.net(obs, return_cache=True)
        mu, log_std = raw.chunk(2, dim=-1)
        merged_block_inputs = {f"actor.{k}": v for k, v in cache.get("block_inputs", {}).items()}
        cache = {"actor": cache, "block_inputs": merged_block_inputs}
        return {"mu": mu, "log_std": log_std, "raw": raw}, cache

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(u)
        return y * self.action_scale + self.action_bias

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.forward(obs)
        mu = out["mu"]
        return self._squash(mu)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retourne (action, log_pi, mean_action). log_pi: [B,1]"""
        out = self.forward(obs)
        mu = out["mu"]
        log_std = out["log_std"]

        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        std = log_std.exp()

        eps = torch.randn_like(mu)
        u = mu + std * eps
        y = torch.tanh(u)

        action = y * self.action_scale + self.action_bias
        mean_action = self._squash(mu)

        log_prob = -0.5 * (eps.pow(2) + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi, device=mu.device)))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        log_det = torch.log(self.action_scale * (1.0 - y.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        log_pi = log_prob - log_det

        return action, log_pi, mean_action


class DoubleQCritic(BlockModel):
    """Double Q-network (Q1, Q2) pour SAC."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = str(activation).lower()

        in_dim = self.obs_dim + self.act_dim
        dims = [in_dim] + [self.hidden_dim] * self.num_layers + [1]
        self.q1 = MLPStack(dims, activation=self.activation)
        self.q2 = MLPStack(dims, activation=self.activation)

        self.q1_linears = self.q1.linears
        self.q2_linears = self.q2.linears

    def get_blocks(self) -> list[BlockSpec]:
        blocks: list[BlockSpec] = []
        for i, lin in enumerate(self.q1_linears):
            is_out = (i == (len(self.q1_linears) - 1))
            name = f"q1.layer{i}" if not is_out else "q1.head"
            blocks.append(BlockSpec(name=name, module=lin, rep="identity", is_output=is_out, group="critic"))
        for i, lin in enumerate(self.q2_linears):
            is_out = (i == (len(self.q2_linears) - 1))
            name = f"q2.layer{i}" if not is_out else "q2.head"
            blocks.append(BlockSpec(name=name, module=lin, rep="identity", is_output=is_out, group="critic"))
        return blocks

    def forward(self, obs_or_x: torch.Tensor, act: torch.Tensor = None, return_cache: bool = False):
        if act is not None:
            x = torch.cat([obs_or_x, act], dim=-1)
        else:
            x = obs_or_x

        if not return_cache:
            q1 = self.q1(x).squeeze(-1)
            q2 = self.q2(x).squeeze(-1)
            return {"q1": q1, "q2": q2}

        q1, c1 = self.q1(x, return_cache=True)
        q2, c2 = self.q2(x, return_cache=True)
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)

        merged_block_inputs = {}
        for k, v in c1.get("block_inputs", {}).items():
            merged_block_inputs[f"q1.{k}"] = v
        for k, v in c2.get("block_inputs", {}).items():
            merged_block_inputs[f"q2.{k}"] = v

        cache = {"q1": c1, "q2": c2, "block_inputs": merged_block_inputs}
        return {"q1": q1, "q2": q2}, cache

    def min_q(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        out = self.forward(obs, act)
        return torch.min(out["q1"], out["q2"])
