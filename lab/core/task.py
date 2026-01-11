# lab/core/task.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Any

class ClassificationTask:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def loss(self, logits, y):
        return F.cross_entropy(logits, y)

    @torch.no_grad()
    def metrics(self, logits, y):
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return {"acc": acc.item()}

class DQNTask:
    """
    q_values: Tensor [B, A]
    y: dict with:
      - "action": LongTensor [B]
      - "target": FloatTensor [B]
    """
    def loss(self, q_values: torch.Tensor, y: Dict[str, Any]) -> torch.Tensor:
        a = y["action"].long()
        target = y["target"].float()
        q_sa = q_values.gather(1, a.view(-1, 1)).squeeze(1)
        return F.smooth_l1_loss(q_sa, target)

    def metrics(self, q_values: torch.Tensor, y: Dict[str, Any]) -> Dict[str, float]:
        with torch.no_grad():
            a = y["action"].long()
            target = y["target"].float()
            q_sa = q_values.gather(1, a.view(-1, 1)).squeeze(1)
            td = (q_sa - target).abs().mean().item()
        return {"td_abs": float(td)}
    
    
# tasks/ppo_task.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


@dataclass
class PPOConfig:
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: Optional[float] = None  # si tu veux early-stop


class PPOTask:
    """
    model_out: {"logits": [B,A], "value": [B]}
    y: dict contenant:
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

        # ratio
        logratio = new_logprobs - old_logprobs
        ratio = logratio.exp()

        # advantage norm
        if self.cfg.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # policy loss (clip)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # value loss (option clip)
        if self.cfg.clip_vloss:
            v_unclipped = (values - returns) ** 2
            v_clipped = old_values + torch.clamp(values - old_values, -self.cfg.clip_coef, self.cfg.clip_coef)
            v_clipped_loss = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
        else:
            v_loss = 0.5 * ((values - returns) ** 2).mean()

        loss = pg_loss - self.cfg.ent_coef * entropy + self.cfg.vf_coef * v_loss

        # stats utiles (pour logs)
        with torch.no_grad():
            approx_kl = (ratio - 1.0 - logratio).mean().item()  # approx KL
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
        # Backprop() appelle metrics(logits,y) après update; on renvoie les derniers stats
        return dict(self._last)
