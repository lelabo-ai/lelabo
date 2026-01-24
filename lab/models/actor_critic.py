# lab/models/actor_critic.py
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .blocks import BlockModel, BlockSpec
    from .mlp_utils import MLPStack
except Exception:  # pragma: no cover
    from blocks import BlockModel, BlockSpec
    from mlp_utils import MLPStack


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
        hidden_dim: int = 256,
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
