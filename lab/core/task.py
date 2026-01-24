# lab/core/task.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# ============================================================
# Utils
# ============================================================

def _onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = torch.zeros(labels.size(0), num_classes, device=labels.device, dtype=torch.float32)
    oh.scatter_(1, labels.view(-1, 1), 1.0)
    return oh


# ============================================================
# Supervised classification
# ============================================================

class ClassificationTask:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, y)

    @torch.no_grad()
    def metrics(self, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        return {"acc": float(acc)}

    @torch.no_grad()
    def output_deltas(self, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return dL/dlogits for CE:
          (softmax - onehot) / B
        """
        B = logits.size(0)
        p = torch.softmax(logits, dim=1)
        oh = _onehot(y.long(), num_classes=logits.size(1)).to(dtype=logits.dtype)
        dlogits = (p - oh) / float(B)
        return {"logits": dlogits}


# ============================================================
# DQN
# ============================================================

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

    @torch.no_grad()
    def output_deltas(self, q_values: torch.Tensor, y: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Return dL/dq_values for SmoothL1 (Huber) on the chosen action only.

        smooth_l1(q_sa, target):
          e = q_sa - target
          grad = e if |e|<1 else sign(e)
        Then average over batch => /B.
        """
        a = y["action"].long()
        target = y["target"].float()

        q_sa = q_values.gather(1, a.view(-1, 1)).squeeze(1)  # [B]
        e = (q_sa - target)  # [B]
        abs_e = e.abs()
        grad_sa = torch.where(abs_e < 1.0, e, e.sign())  # [B]

        B = q_values.size(0)
        grad_sa = grad_sa / float(B)

        dQ = torch.zeros_like(q_values)
        dQ.scatter_(1, a.view(-1, 1), grad_sa.view(-1, 1))
        return {"q_values": dQ}


# ============================================================
# PPO
# ============================================================

@dataclass
class PPOConfig:
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: Optional[float] = None  # if you want early-stop


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
        return dict(self._last)

    @torch.no_grad()
    def output_deltas(self, model_out: Dict[str, torch.Tensor], y: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Return output deltas compatible with THIS PPO loss (clip + entropy + vf).

        Returns:
          - "logits": dL/dlogits  [B, A]
          - "value":  dL/dvalue   [B, 1]
        """
        logits = model_out["logits"]
        values = model_out["value"]
        if values.dim() == 1:
            values_ = values.view(-1, 1)
        else:
            values_ = values

        actions = y["actions"].long()
        old_logprobs = y["old_logprobs"].float()
        advantages = y["advantages"].float()
        returns = y["returns"].float()
        old_values = y["old_values"].float()

        B = logits.size(0)

        # distribution
        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(actions)  # [B]
        logratio = new_logprobs - old_logprobs
        ratio = logratio.exp()

        # advantage norm
        if self.cfg.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # -------------------------
        # Policy part: pg_loss = mean(max(pg1, pg2))
        # pg1 = -adv * r
        # pg2 = -adv * clamp(r, 1-e, 1+e)
        # d(pg1)/d logp = -adv * r
        # d(pg2)/d logp = -adv * r if r in clip-range else 0
        # choose branch by pg1 > pg2
        # then average => /B
        # -------------------------
        e = float(self.cfg.clip_coef)
        pg1 = -advantages * ratio
        r_clamped = torch.clamp(ratio, 1.0 - e, 1.0 + e)
        pg2 = -advantages * r_clamped

        use_pg1 = (pg1 > pg2)
        in_clip = (ratio >= (1.0 - e)) & (ratio <= (1.0 + e))

        d_pg1_dlogp = -advantages * ratio
        d_pg2_dlogp = torch.where(in_clip, -advantages * ratio, torch.zeros_like(ratio))

        dL_dlogp = torch.where(use_pg1, d_pg1_dlogp, d_pg2_dlogp) / float(B)  # [B]

        # convert dL/dlogp to dL/dlogits: d logpi(a)/d logits = onehot(a) - pi
        pi = torch.softmax(logits, dim=1)
        onehot_a = torch.zeros_like(pi)
        onehot_a.scatter_(1, actions.view(-1, 1), 1.0)
        delta_logits_pg = dL_dlogp.view(-1, 1) * (onehot_a - pi)  # [B,A]

        # -------------------------
        # Entropy part: loss has - ent_coef * mean(entropy)
        # dL/dz = -ent_coef * (1/B) * dH/dz
        # where dH/dz_j = p_j * (S - (log p_j + 1)), S = sum_k p_k (log p_k + 1)
        # -------------------------
        if self.cfg.ent_coef != 0.0:
            logp = torch.log(pi + 1e-8)
            S = (pi * (logp + 1.0)).sum(dim=1, keepdim=True)  # [B,1]
            dH_dz = pi * (S - (logp + 1.0))                   # [B,A]
            delta_logits_ent = (-float(self.cfg.ent_coef) / float(B)) * dH_dz
        else:
            delta_logits_ent = torch.zeros_like(delta_logits_pg)

        delta_logits = delta_logits_pg + delta_logits_ent  # [B,A]

        # -------------------------
        # Value part: v_loss = 0.5 * mean(max(unclipped, clipped))
        # total adds + vf_coef * v_loss
        # derivative depends on which branch is max
        # -------------------------
        returns_ = returns.view(-1, 1).to(values_.dtype)
        old_values_ = old_values.view(-1, 1).to(values_.dtype)

        if self.cfg.clip_vloss:
            v_unclipped = (values_ - returns_) ** 2
            v_clipped = old_values_ + torch.clamp(values_ - old_values_, -e, e)
            v_clipped_loss = (v_clipped - returns_) ** 2

            use_unclipped = (v_unclipped >= v_clipped_loss)

            # d(0.5*(v-ret)^2)/dv = (v-ret)
            d_unclipped = (values_ - returns_)
            # clipped path: derivative wrt values is 0 if clamp saturated, else 1
            unclamped = (values_ - old_values_).abs() <= e
            d_clipped = (v_clipped - returns_) * unclamped.to(values_.dtype)

            d_v = torch.where(use_unclipped, d_unclipped, d_clipped)  # [B,1]
        else:
            d_v = (values_ - returns_)

        # mean => /B, and vf_coef
        delta_value = (float(self.cfg.vf_coef) / float(B)) * d_v  # [B,1]

        return {"logits": delta_logits, "value": delta_value}
