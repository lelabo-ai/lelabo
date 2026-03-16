# Research Notes

Design choices, current limitations, and future direction. This page is for people who want to understand why LeLabo is built the way it is, not just how to use it.

---

## Design choices

### Plain `nn.Module` models

Models in LeLabo are standard PyTorch `nn.Module` objects. There is no mandatory framework-specific base class.

This keeps models close to how researchers already write PyTorch — no framework overhead, no framework lock-in, and no friction when adapting published code. A model from a paper repository can be registered in LeLabo with minimal changes.

The one voluntary addition is `declare_blocks()`, which a model can implement to expose a semantic block structure for local rules. It is an opt-in protocol, not a requirement.

### Registry-based composition

Experiments are composed from registries: dataset, model, initializer, loss, update rule, optimizer, scheduler, callbacks, metrics.

Each registry entry is a named builder function. The config selects builders by name. This makes ablations cheap and explicit — swapping a component is changing a name in a config, not editing code.

It also means the runtime does not need to know what the components do — just that they satisfy the right contracts.

### Cache-first for local rules

Local learning rules need intermediate activations. The common alternative is to write a custom training loop per rule, which makes it hard to compare rules under the same runtime conditions.

LeLabo's approach: a stable cache contract (`declare_blocks`, `CacheSpec`, `forward_with_standard_cache`) that any rule can use without modifying the training loop or the model class. The rule declares what it needs; the runtime collects it.

This is not perfect — some papers require things that don't map cleanly to this contract. But for the majority of local-rule setups, it works.

### CLI-first, Python API second

The primary interface is the command line. Python-level usage is supported but not the main story.

This is intentional: it makes experiments easier to script, log, and reproduce. A run is a command, not a script. The resolved config captures exactly what ran.

---

## Current limitations

### RL is not mature yet

RL support exists — there are algorithms, runners, and configs. But it is not as stable or as well-documented as the supervised stack. The RL surface is under active development.

See the [RL page](rl.md) for the current status.

### No staged multi-phase programs as first-class objects

Some papers require explicit phase transitions: pretrain, then freeze, then fine-tune. Or multiple loaders and multiple optimizers at different stages.

LeLabo does not yet provide a first-class abstraction for this. Some setups can be implemented with workarounds, but the runtime does not have a dedicated staged-program primitive.

This is likely the next major conceptual addition after the current supervised and capsule surfaces stabilize.

### Public APIs are still evolving

LeLabo is at v0.1.0. The extension points documented in the [Public API](reference/api.md) are intentional and treated as stable. Everything else — internal helpers, private methods, cache reconstruction internals — can change.

---

## Future direction

The clearest next step is a stronger abstraction for staged training programs. This would let the runtime natively express:

- pretrain + fine-tune workflows
- multiple loaders with explicit phase transitions
- multiple optimizers over different parameter groups at different stages

Beyond that, the longer-term direction is the capsule ecosystem: making it easier to package methods as reusable capsules, share them, and compare them under consistent runtime conditions.
