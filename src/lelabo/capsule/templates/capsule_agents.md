# Capsule Extension Guide

This file is the entry point for this capsule.

Use it if you are:

- a researcher extending LeLabo quickly
- an agent such as Codex/Claude working from the capsule files directly

## Official first path

The official first path in this scaffold is:

1. Uncomment `@register_optimizer("capsule_sgd")` in `optimizers/example.py`
2. Run `lelabo list optimizers`
3. Run `lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml`
4. Run `pytest -q tests`

## Open these files in this order

1. `README.md`
2. `AGENTS.md`
3. `resources/LELABO_REFERENCE.md`
4. `resources/PARAM_FLOW.md`
5. `resources/MODEL_CACHE_ADVANCED.md` only if you are working on cache-aware models or local rules
6. `resources/UPDATE_RULE_LIFECYCLE.md` only if you are implementing a custom update rule
7. `resources/PAPER_PACK_PLAYBOOK.md` only if you are implementing a full paper pack

## If your idea is X, edit Y

- New architecture or model: `models/`
- New local learning rule: `update_rules/`
- New dataset loader or split logic: `datasets/`
- New train/eval metric: `metrics/`
- New parameter initializer: `initializers/`
- New loss: `losses/`
- New optimizer: `optimizers/`
- New scheduler: `schedulers/`
- New trainer callback: `callbacks/`
- Full paper implementation: start with `resources/PAPER_PACK_PLAYBOOK.md`

## Directory map

- `models/`
  Regular model plugins. Start with `models/example.py`.
  If you need cache-aware blocks for local rules, read
  `resources/MODEL_CACHE_ADVANCED.md` and `models/cache_walkthrough.py`.

- `update_rules/`
  Learning rules such as local rules or alternative credit assignment methods.
  If you are implementing one, also read `resources/UPDATE_RULE_LIFECYCLE.md`.

- `configs/`
  Example configs.
  Start with `configs/train.supervised.capsule_optimizer.toml`.
  If you are implementing a paper pack, read `configs/train.supervised.paper_pack.toml`.

- `tests/`
  Local smoke tests for the capsule.
  Start with `tests/test_capsule_optimizer_smoke.py`.
  For multi-component work, also read `tests/test_paper_pack_smoke.py`.

- `resources/`
  Local contract/reference docs for agents and researchers.

## Core contracts

Use `resources/LELABO_REFERENCE.md` for:

- builder signatures
- context fields
- `DataBundle`
- discovery rules
- useful public helpers

Use `resources/PARAM_FLOW.md` for:

- TOML -> runtime param mapping

Use `resources/MODEL_CACHE_ADVANCED.md` for:

- `declare_blocks()`
- `BlockSpec`
- `ResolvedBlock`
- `CacheSpec`
- cache-aware models

Use `resources/UPDATE_RULE_LIFECYCLE.md` for:

- what an update rule may do
- what it must return
- current runtime limits

Use `resources/PAPER_PACK_PLAYBOOK.md` for:

- model + update_rule + dataset + config + test decomposition

Registry contract shortcuts:

- `models/`: `@register_model(name)` then `build_xxx(ctx, args) -> nn.Module`
- `update_rules/`: `@register_update_rule(name)` then builder -> `UpdateRule`
- `datasets/`: `@register_dataset(name)` then builder -> `DataBundle`
- `metrics/`: `@register_metric(name, kind=...)` then streaming metric class
- `initializers/`: `@register_initializer(name)` then builder -> callable
- `losses/`: `@register_loss(name)` then builder -> callable
- `optimizers/`: `@register_optimizer(name)` then builder -> `torch.optim.Optimizer`
- `schedulers/`: `@register_scheduler(name)` then builder -> scheduler/controller
- `callbacks/`: `@register_callback(name)` then builder -> callback

## Validation commands

Use these commands from the capsule root:

```bash
pytest -q tests
```

```bash
lelabo list optimizers
```

```bash
lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml
```

Useful follow-up commands:

```bash
lelabo list models
```

```bash
lelabo list update-rules
```

```bash
lelabo train supervised --config configs/train.supervised.paper_pack.toml
```

## Common failure modes

- Plugin does not appear in `lelabo list`
  The `@register_*` line is still commented, or the file has an import error.

- Model/update rule works in BP but fails for local rules
  Read `resources/MODEL_CACHE_ADVANCED.md`.

- Update rule contract is unclear
  Read `resources/UPDATE_RULE_LIFECYCLE.md`.

- The extension spans several components and starts feeling ad hoc
  Read `resources/PAPER_PACK_PLAYBOOK.md`.

## Mini prompt recipe for an agent

You can give an agent a prompt like:

> Open `AGENTS.md`, follow the capsule contracts, implement my component in the right folder, keep only the decorator commented until the code is ready, then tell me which `lelabo list ...` and `pytest -q tests` commands to run.
