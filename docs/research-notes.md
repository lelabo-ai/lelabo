# Research Notes

## Project posture

LeLabo is still a research workbench.

The current priority is:

- strong supervised workflows
- explicit local-rule support
- low-friction extension through capsules

## Current design choices

### Plain `nn.Module` models

LeLabo keeps models close to ordinary PyTorch:

- regular `forward(...)`
- standard submodules
- no mandatory framework-specific model base class for common cases

### Registry-based composition

Experiments are composed from registries:

- dataset
- model
- initializer
- loss
- update rule
- optimizer
- scheduler
- callbacks

This keeps ablations cheap and explicit.

### Cache-first support for local rules

Local rules are supported through a common cache contract rather than through model-specific training loops.

That contract currently revolves around:

- `declare_blocks()`
- `BlockSpec`
- `ResolvedBlock`
- `CacheSpec`
- `forward_with_standard_cache(...)`

## Current limitations

- public APIs are still evolving
- RL is not documented as deeply as supervised in this phase
- staged multi-phase experiment programs are not first-class runtime objects yet
- some papers with multiple loaders, phases, or optimizers still require workaround logic

## Future direction

The likely next conceptual step after this phase is a stronger runtime abstraction for staged training programs.

That would help represent papers that need:

- pretrain then fine-tune
- several loaders
- several optimizers
- explicit phase transitions
