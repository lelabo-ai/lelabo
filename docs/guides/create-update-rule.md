# Create an Update Rule

Register a custom learning rule that the trainer will use instead of standard backpropagation.

## What an update rule actually is

An update rule is the general control point for learning. `train_step` receives a batch and a state object, and from there you write whatever logic you want — no constraints on what you compute, which tensors you access, or how you apply updates.

Standard backpropagation is just one instantiation of this interface. But you can use it for much more:

- **Alternative credit assignment** — DFA, FA, DNI, local rules, biologically inspired updates
- **Per-layer optimizers** — different learning rates, different optimizer types for different parts of the model
- **Architectural constraints** — impose geometric constraints, weight symmetry, orthogonality, or any structural invariant during training
- **Asymmetric or conditional updates** — freeze certain layers, update different parts on different steps, apply updates conditionally based on activations
- **Custom loss surfaces** — combine multiple objectives, contrastive terms, auxiliary losses, custom regularizers
- **Any custom learning logic** — if it can be expressed as "given this batch, do something to the model", it belongs here

The key idea: you control exactly **what you give**, **how you give it**, and **to whom** at every step.

## When to use this

Use an update rule extension whenever you need control over the learning loop that goes beyond choosing a standard optimizer or loss. If the standard BP + optimizer path doesn't give you what you need, the update rule is where to implement it.

If you only need a custom optimizer within standard BP, use an optimizer extension instead (see [Create a capsule](create-capsule.md)).

## Where to edit

Inside your capsule: `update_rules/example.py`

## Builder shape

```python
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule


@register_update_rule("my_rule")
def build_my_rule(ctx: UpdateRuleContext):
    params = ctx.extra.get("update_rule_params", {}) or {}
    grad_clip = params.get("grad_clip", 1.0)

    class MyRule:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer
            self.grad_clip = grad_clip

        def train_step(self, batch, state):
            x, y = batch
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = ctx.loss_fn(out, y)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            self.optimizer.step()
            return {"loss": loss.item()}

    return MyRule
```

!!! note "`train_step` must return numeric scalars"
    The return dict from `train_step` must contain only numeric scalars — no tensors. The trainer uses these values for logging and monitoring.

## Where params come from

- Primary source: `ctx.extra["update_rule_params"]`
- Optional mirror: `args.update_rule_params`

Always read from `ctx.extra["update_rule_params"]`, not from a hardcoded default in the builder.

## Config snippet

```toml
[update_rule]
name = "my_rule"

[update_rule.params]
grad_clip = 1.0
```

## Verify

```bash
lelabo list update-rules
# my_rule should appear

lelabo train supervised \
  --config configs/train/supervised.paper_pack.toml \
  --rule my_rule
```

## Local rule with cache

If your rule needs intermediate activations, use `forward_with_standard_cache()`:

```python
import torch
import torch.nn as nn
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule
from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache


@register_update_rule("my_local_rule")
def build_my_local_rule(ctx: UpdateRuleContext):
    cache_spec = CacheSpec(
        target_view="execution",
        trainable_module_types=(nn.Linear,),
        capture_inputs=True,
        capture_outputs=True,
        require_single_call=True,
    )

    class MyLocalRule:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer

        def train_step(self, batch, state):
            x, y = batch
            self.optimizer.zero_grad()

            out, cache, blocks = forward_with_standard_cache(
                self.model, x, cache_spec=cache_spec
            )

            # blocks is a list of ResolvedBlock
            # use block.x, block.h to compute local updates
            loss = ctx.loss_fn(out, y)

            # ... compute and apply local updates ...

            return {"loss": loss.item()}

    return MyLocalRule
```

See [Cache & local rules](../concepts/cache-local-rules.md) for the full contract.

## Common mistakes

- Returning tensors instead of scalars from `train_step`
- Assuming multiple cache views are available simultaneously
- Reading rule params from the wrong source
- Depending on block structure the model does not expose
