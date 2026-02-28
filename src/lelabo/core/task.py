# lab/core/task.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from ..metrics.payload import build_metric_payload


# ============================================================
# Utils
# ============================================================

def _onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = torch.zeros(labels.size(0), num_classes, device=labels.device, dtype=torch.float32)
    oh.scatter_(1, labels.view(-1, 1), 1.0)
    return oh


def _classification_targets(y: torch.Tensor, *, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    if y.dim() == 2 and int(y.size(1)) == 1 and not torch.is_floating_point(y):
        y = y.view(-1)
    if y.dim() == 1:
        return _onehot(y.long(), num_classes=num_classes).to(dtype=dtype)
    if y.dim() == 2 and int(y.size(1)) == int(num_classes):
        return y.to(dtype=dtype)
    raise ValueError(
        "Expected class targets shaped [B], [B,1], or [B,C]; "
        f"got {tuple(y.shape)} with C={int(num_classes)}."
    )


def _as_class_indices(y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 2 and int(y.size(1)) == 1 and not torch.is_floating_point(y):
        return y.view(-1).long()
    if y.dim() == 2 and torch.is_floating_point(y):
        return y.argmax(dim=1).long()
    if y.dim() == 1:
        return y.long()
    raise ValueError(f"Expected labels shaped [B], [B,1], or one-hot [B,C], got {tuple(y.shape)}.")


def _loss_reduction(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None) -> str:
    module = getattr(loss_fn, "module", None)
    reduction = getattr(module, "reduction", "mean")
    return str(reduction).strip().lower()


def _apply_reduction_to_delta(delta: torch.Tensor, reduction: str) -> torch.Tensor:
    red = str(reduction).strip().lower()
    if red == "mean":
        denom = max(1, int(delta.numel()))
        return delta / float(denom)
    if red in {"sum", "none"}:
        return delta
    raise ValueError(f"Unsupported loss reduction '{reduction}'.")


# ============================================================
# Supervised classification
# ============================================================

class ClassificationTask:
    def __init__(
        self,
        num_classes: int,
        *,
        loss_name: str = "cross_entropy",
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        self.num_classes = int(num_classes)
        self.loss_name = str(loss_name).strip().lower()
        if loss_fn is None:
            self.loss_fn = lambda logits, y: F.cross_entropy(logits, _as_class_indices(y))
        else:
            self.loss_fn = loss_fn

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, y)

    @torch.no_grad()
    def metrics(self, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        y_idx = _as_class_indices(y)
        preds = logits.argmax(dim=1)
        acc = (preds == y_idx).float().mean().item()
        out: Dict[str, Any] = {"acc": float(acc)}
        out.update(
            build_metric_payload(
                y_true=y_idx,
                y_pred=preds,
                kind="classification",
            )
        )
        return out

    @torch.no_grad()
    def output_deltas(self, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        name = self.loss_name
        reduction = _loss_reduction(self.loss_fn)
        num_classes = int(logits.size(1))
        targets = _classification_targets(
            y,
            num_classes=num_classes,
            dtype=logits.dtype,
        ).to(device=logits.device)

        if name in {"cross_entropy", "ce"}:
            module = getattr(self.loss_fn, "module", None)
            if module is not None:
                weight = getattr(module, "weight", None)
                ignore_index = int(getattr(module, "ignore_index", -100))
                if weight is not None or ignore_index != -100:
                    raise NotImplementedError(
                        "output_deltas for cross_entropy supports only default weight/ignore_index."
                    )
                smooth = float(getattr(module, "label_smoothing", 0.0))
            else:
                smooth = 0.0
            if smooth > 0.0:
                targets = targets * (1.0 - smooth) + (smooth / float(max(1, num_classes)))
            delta = torch.softmax(logits, dim=1) - targets
            batch = max(1, int(logits.size(0)))
            if reduction == "mean":
                delta = delta / float(batch)
            elif reduction not in {"sum", "none"}:
                raise ValueError(f"Unsupported cross-entropy reduction '{reduction}'.")
            return {"logits": delta}

        if name in {"bce_with_logits", "bce_logits"}:
            module = getattr(self.loss_fn, "module", None)
            if module is not None and (
                getattr(module, "weight", None) is not None
                or getattr(module, "pos_weight", None) is not None
            ):
                raise NotImplementedError(
                    "output_deltas for bce_with_logits supports only default weight/pos_weight."
                )
            delta = torch.sigmoid(logits) - targets
            return {"logits": _apply_reduction_to_delta(delta, reduction)}

        if name in {"bce", "binary_cross_entropy"}:
            module = getattr(self.loss_fn, "module", None)
            if module is not None and getattr(module, "weight", None) is not None:
                raise NotImplementedError("output_deltas for bce supports only default weights.")
            eps = torch.finfo(logits.dtype).eps
            probs = logits.clamp(min=eps, max=1.0 - eps)
            delta = (probs - targets) / (probs * (1.0 - probs))
            return {"logits": _apply_reduction_to_delta(delta, reduction)}

        if name in {"mse", "mse_loss"}:
            if torch.is_tensor(y) and y.dim() == logits.dim():
                target = y.to(device=logits.device, dtype=logits.dtype)
            else:
                target = targets
            delta = 2.0 * (logits - target)
            return {"logits": _apply_reduction_to_delta(delta, reduction)}

        raise NotImplementedError(
            f"output_deltas is not implemented for loss '{self.loss_name}'."
        )


class GLUETask:
    """
    Task helper for HuggingFace sequence classification/regression batches.
    """

    def __init__(self, task_name: str, is_regression: bool, num_labels: int):
        self.task_name = str(task_name).lower()
        self.is_regression = bool(is_regression)
        self.num_labels = int(num_labels)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.is_regression:
            preds = logits.view(-1).float()
            targets = y.view(-1).float()
            return F.mse_loss(preds, targets)
        return F.cross_entropy(logits, y.long())

    @torch.no_grad()
    def metrics(self, outputs_or_logits: Any, y: torch.Tensor) -> Dict[str, Any]:
        logits = outputs_or_logits.logits if hasattr(outputs_or_logits, "logits") else outputs_or_logits
        if not torch.is_tensor(logits):
            return {}
        if self.is_regression:
            preds = logits.view(-1).float()
            targets = y.view(-1).float()
            mse = F.mse_loss(preds, targets).item()
            mae = F.l1_loss(preds, targets).item()
            out: Dict[str, Any] = {"mse": float(mse), "mae": float(mae), "metric": float(-mse)}
            out.update(
                build_metric_payload(
                    y_true=targets,
                    y_pred=preds,
                    kind="regression",
                )
            )
            return out

        preds = logits.argmax(dim=-1)
        acc = (preds == y.long()).float().mean().item()
        out = {"acc": float(acc), "metric": float(acc)}
        out.update(
            build_metric_payload(
                y_true=y.long(),
                y_pred=preds,
                kind="classification",
            )
        )
        return out

    @torch.no_grad()
    def evaluate(self, model, loader, device: str) -> Dict[str, float]:
        total_n = 0
        total_loss = 0.0
        total_acc = 0.0
        total_mse = 0.0
        total_mae = 0.0

        for batch in loader:
            if not isinstance(batch, dict):
                raise TypeError("GLUETask expects mapping batches with HuggingFace collate output.")

            b: Dict[str, Any] = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    b[k] = v.to(device)
                else:
                    b[k] = v

            outputs = model(**b)
            labels = b["labels"]
            logits = outputs.logits
            loss = outputs.loss if getattr(outputs, "loss", None) is not None else self.loss(logits, labels)

            n = int(labels.shape[0]) if hasattr(labels, "shape") and labels.shape else 1
            w = float(max(1, n))
            total_n += int(w)
            total_loss += float(loss.item()) * w

            stats = self.metrics(outputs, labels)
            if self.is_regression:
                total_mse += float(stats.get("mse", 0.0)) * w
                total_mae += float(stats.get("mae", 0.0)) * w
            else:
                total_acc += float(stats.get("acc", 0.0)) * w

        denom = float(max(1, total_n))
        out: Dict[str, float] = {"loss": float(total_loss / denom)}
        if self.is_regression:
            mse = float(total_mse / denom)
            out["mse"] = mse
            out["mae"] = float(total_mae / denom)
            out["metric"] = float(-mse)
        else:
            acc = float(total_acc / denom)
            out["acc"] = acc
            out["metric"] = acc
        return out


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


class SACCriticTask:
    """Loss pour DoubleQCritic.
    model_out: {"q1": [B], "q2": [B]}
    y: {"target": [B]}
    """
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
        return dict(self._last)


class SACActorTask:
    """Loss actor SAC (reparam + tanh squash) avec un DoubleQCritic externe.
    model_out: {"mu": [B,A], "log_std": [B,A]}
    y: {"obs": [B,obs_dim]}
    """
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
        u = mu + std * eps
        y_t = torch.tanh(u)
        action = y_t * self.action_scale + self.action_bias

        log_prob = -0.5 * (eps.pow(2) + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi, device=mu.device)))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_det = torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
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
        return dict(self._last)
