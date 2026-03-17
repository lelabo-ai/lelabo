# Implement a Method with an AI Assistant

This tutorial shows how to use an AI coding agent (Claude Code, Codex, Cursor, etc.) to implement a research method inside a LeLabo capsule. The capsule's local documentation is designed to make agents effective — they read the contracts and recipes automatically, so you focus on describing your idea rather than explaining the framework.

We'll implement a custom update rule: **Noise-Scaled Feedback Alignment** — a made-up variant of Feedback Alignment where the feedback matrix is scaled by a noise schedule that decays over training. Simple enough to implement in one session, interesting enough to show the workflow.

## Step 1 — Create the capsule

```bash
lelabo capsule init nsfa_method
cd nsfa_method
```

## Step 2 — Open in your IDE with an AI assistant

Open the capsule folder in VS Code, Cursor, or any IDE with an AI coding assistant. The agent will see the capsule structure and the `resources/` documentation.

<!-- screenshot: IDE open with capsule structure visible in sidebar -->

## Step 3 — Describe your idea to the agent

Here's the prompt to give your agent:

```
I want to implement a custom update rule called "nsfa" (Noise-Scaled Feedback Alignment).

The idea: it's like standard Feedback Alignment, but the random feedback
matrices are scaled by a noise factor that starts high and decays over
training. Early in training, the feedback signal is noisy (encouraging
exploration). Later, it becomes cleaner (encouraging convergence).

The noise schedule: noise_scale = initial_noise * (decay_rate ^ epoch)

Parameters I want exposed in the config:
- initial_noise: float, default 1.0
- decay_rate: float, default 0.95
- feedback_scale: float, default 1.0

It should work with any model that implements declare_blocks().
Use the MLP builtin for testing, with MNIST.

Please implement:
1. The update rule in update_rules/example.py
2. A config in configs/ that runs it on MNIST
3. A smoke test in tests/
```

<!-- screenshot: the prompt in Claude Code / Cursor -->

## Step 4 — The agent reads the contracts

When the agent starts working, it will read the capsule's local documentation automatically:

- `resources/EXTENSION_RECIPES.md` — step-by-step recipes for each extension type
- `resources/LELABO_REFERENCE.md` — exact contracts, field names, return types
- `resources/UPDATE_RULE_LIFECYCLE.md` — how update rules are built, called, and torn down
- `resources/PARAM_FLOW.md` — how parameters flow from the TOML config to the builder

These files give the agent precise context about:

- What signature the builder function needs
- Where to read rule params (`ctx.extra.get("update_rule_params", {})`)
- How `CacheSpec` and `forward_with_standard_cache()` work
- What `train_step` must return (dict of numeric scalars)
- How to structure the TOML config

<!-- screenshot: agent reading resources/ files -->

## Step 5 — Review the implementation

The agent should produce something close to this:

```python
# update_rules/example.py
import torch
import torch.nn as nn
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule
from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache


@register_update_rule("nsfa")
def build_nsfa(ctx: UpdateRuleContext):
    params = ctx.extra.get("update_rule_params", {}) or {}
    initial_noise = params.get("initial_noise", 1.0)
    decay_rate = params.get("decay_rate", 0.95)
    feedback_scale = params.get("feedback_scale", 1.0)

    cache_spec = CacheSpec(
        target_view="execution",
        trainable_module_types=(nn.Linear,),
        capture_inputs=True,
        capture_outputs=True,
        require_single_call=True,
    )

    class NSFA:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer
            self.feedback = {}
            self.current_epoch = 0

        @property
        def noise_scale(self):
            return initial_noise * (decay_rate ** self.current_epoch)

        def train_step(self, batch, state):
            if state is not None:
                self.current_epoch = getattr(state, "epoch", 0)

            x, y = batch
            self.optimizer.zero_grad()

            out, _cache, views = forward_with_standard_cache(
                self.model, x, cache_spec=cache_spec
            )
            blocks = views.get("execution", [])

            loss = ctx.loss_fn(out, y)

            # ... feedback alignment logic with noise scaling ...
            # The agent fills in the full implementation here

            self.optimizer.step()

            return {
                "loss": loss.item(),
                "noise_scale": self.noise_scale,
            }

    return NSFA
```

**What to check in the review:**

- Does the builder read params from `ctx.extra.get("update_rule_params", {})`? (not hardcoded)
- Does `train_step` return only numeric scalars? (no tensors)
- Does the `CacheSpec` match what the rule needs?
- Is the noise schedule applied correctly to the feedback matrices?
- Does the config point to the right component names?

<!-- screenshot: reviewing the generated code -->

## Step 6 — Run and test

```bash
# Verify registration
lelabo list update-rules

# Run
lelabo train supervised --config configs/train/supervised.paper_pack.toml

# Run the smoke test
pytest -q tests/
```

<!-- screenshot: successful run output -->

## Why this works

The capsule's `resources/` documentation acts as a contract specification for the agent. Instead of guessing how LeLabo works, the agent reads:

- The exact function signatures it needs to implement
- The exact fields it can access on context objects
- The exact return type expectations
- Working code examples for each extension type

This is not magic — it's documentation doing what documentation should do. The agent is effective because the contracts are explicit and machine-readable.

## When it works less well

The agent may struggle with:

- **Very non-standard setups** — if your method doesn't fit the cache/block contract, the agent may try to force it
- **Multi-component methods** — implementing a model + rule + dataset + custom metric simultaneously can lead to inconsistencies
- **Performance-sensitive code** — the agent will produce correct code, but not necessarily optimized code

In these cases, implement the tricky parts yourself and let the agent handle the boilerplate.

## Help us improve the agent experience

If you use an AI agent with LeLabo and find that:

- The `resources/` docs are unclear or incomplete for a specific use case
- The agent consistently makes the same mistake
- A contract is underspecified and the agent guesses wrong

**[Open an issue](https://github.com/adrienkegreisz/LeLabo/issues)** and describe what happened. This feedback directly improves the local documentation for everyone — including future agents working on other capsules.

## Next steps

- [Create a capsule](../guides/create-capsule.md) — the capsule workflow without an agent
- [Create an update rule](../guides/create-update-rule.md) — manual implementation guide
- [Community](../community.md) — contributing and sharing capsules
