# Custom Training Logic

The `train_step` in an update rule is a general control point — you decide what happens at every training step. This tutorial shows three non-standard training patterns that go beyond simple backpropagation.

## Example 1 — Per-layer optimizer

Different learning rates for different parts of the model. A common pattern when fine-tuning: slow learning for pretrained layers, fast learning for the head.

```python
import torch
import torch.nn as nn
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule


@register_update_rule("per_layer_lr")
def build_per_layer_lr(ctx: UpdateRuleContext):
    params = ctx.extra.get("update_rule_params", {}) or {}
    head_lr_mult = params.get("head_lr_mult", 10.0)

    class PerLayerLR:
        def __init__(self, model, optimizer):
            self.model = model
            # Split parameters into "body" and "head"
            head_params = []
            body_params = []
            for name, param in model.named_parameters():
                if "output" in name or "head" in name or "classifier" in name:
                    head_params.append(param)
                else:
                    body_params.append(param)

            base_lr = optimizer.defaults["lr"]
            self.optimizer = torch.optim.Adam([
                {"params": body_params, "lr": base_lr},
                {"params": head_params, "lr": base_lr * head_lr_mult},
            ])

        def train_step(self, batch, state):
            x, y = batch
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = ctx.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item()}

    return PerLayerLR
```

Config:

```toml
[update_rule]
name = "per_layer_lr"
[update_rule.params]
head_lr_mult = 10.0
```

## Example 2 — Progressive layer freeze

Freeze early layers after a warmup period, then only train the later layers. Useful for preventing catastrophic forgetting or stabilizing training.

```python
import torch
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule


@register_update_rule("progressive_freeze")
def build_progressive_freeze(ctx: UpdateRuleContext):
    params = ctx.extra.get("update_rule_params", {}) or {}
    freeze_after_epoch = params.get("freeze_after_epoch", 5)
    freeze_fraction = params.get("freeze_fraction", 0.5)

    class ProgressiveFreeze:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer
            self.all_params = list(model.named_parameters())
            self.frozen = False

        def _freeze_early_layers(self):
            """Freeze the first `freeze_fraction` of parameters."""
            n_freeze = int(len(self.all_params) * freeze_fraction)
            for i, (name, param) in enumerate(self.all_params):
                if i < n_freeze:
                    param.requires_grad_(False)
            self.frozen = True

        def train_step(self, batch, state):
            # Check if we should freeze
            current_epoch = getattr(state, "epoch", 0) if state else 0
            if not self.frozen and current_epoch >= freeze_after_epoch:
                self._freeze_early_layers()

            x, y = batch
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = ctx.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item()}

    return ProgressiveFreeze
```

Config:

```toml
[update_rule]
name = "progressive_freeze"
[update_rule.params]
freeze_after_epoch = 5
freeze_fraction = 0.5
```

This trains all layers for 5 epochs, then freezes the first half and continues training only the rest.

## Example 3 — Auxiliary loss

Add a regularization term alongside the main loss. Here: an orthogonality penalty that encourages weight matrices to have orthogonal rows.

```python
import torch
import torch.nn as nn
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule


@register_update_rule("ortho_reg")
def build_ortho_reg(ctx: UpdateRuleContext):
    params = ctx.extra.get("update_rule_params", {}) or {}
    ortho_weight = params.get("ortho_weight", 0.01)

    class OrthoReg:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer

        def _orthogonality_penalty(self):
            """Penalize deviation from orthogonality in weight matrices."""
            penalty = 0.0
            for module in self.model.modules():
                if isinstance(module, nn.Linear) and module.weight.shape[0] > 1:
                    W = module.weight
                    WWT = W @ W.T
                    identity = torch.eye(W.shape[0], device=W.device)
                    penalty += ((WWT - identity) ** 2).sum()
            return penalty

        def train_step(self, batch, state):
            x, y = batch
            self.optimizer.zero_grad()
            out = self.model(x)

            main_loss = ctx.loss_fn(out, y)
            ortho_loss = self._orthogonality_penalty()
            total_loss = main_loss + ortho_weight * ortho_loss

            total_loss.backward()
            self.optimizer.step()

            return {
                "loss": main_loss.item(),
                "ortho_loss": ortho_loss.item(),
                "total_loss": total_loss.item(),
            }

    return OrthoReg
```

Config:

```toml
[update_rule]
name = "ortho_reg"
[update_rule.params]
ortho_weight = 0.01
```

The extra scalars (`ortho_loss`, `total_loss`) will appear in `metrics.jsonl` alongside the standard metrics.

## Pattern: combining multiple techniques

These patterns compose. You can build a rule that does per-layer optimization, progressive freezing, and custom regularization — all in the same `train_step`. The rule is your single entry point for all training logic.

```python
def train_step(self, batch, state):
    # 1. Check epoch for freezing
    if should_freeze(state):
        freeze_layers()

    # 2. Forward + loss
    out = self.model(x)
    loss = compute_loss(out, y)
    loss += auxiliary_penalty()

    # 3. Backward + step with per-layer optimizer
    loss.backward()
    self.optimizer.step()

    return {"loss": loss.item(), ...}
```

## Next steps

- [Create an update rule](../guides/create-update-rule.md) — the concise reference for update rule contracts
- [Supervised runtime](../concepts/supervised-runtime.md) — understand what the trainer provides to `train_step`
