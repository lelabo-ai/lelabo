# Capsule Extension Guide

This file is the entry point for this capsule.

Use it if you are:

- a researcher extending LeLabo quickly
- an agent such as Codex/Claude working from the capsule files directly

## What a capsule is for

A capsule is a local extension workspace. It lets you add custom LeLabo components
without modifying the main library.

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

## Directory map

- `models/`
  Regular model plugins. Start with `models/example.py`.
  If you need advanced cache/block patterns for local rules, read
  `resources/MODEL_CACHE_ADVANCED.md` and `models/cache_walkthrough.py`.

- `update_rules/`
  Learning rules such as local rules or alternative credit assignment methods.

- `datasets/`
  Dataset builders returning a `DataBundle`.

- `metrics/`
  Trainer metrics. Prefer streaming metrics.

- `initializers/`
  Builders returning callables that initialize a model in-place.

- `losses/`
  Builders returning the loss callable used by the trainer.

- `optimizers/`
  Optimizer builders. This is the official first extension path in this capsule.

- `schedulers/`
  Scheduler builders.

- `callbacks/`
  Trainer callbacks.

- `configs/`
  Example configs. Start with `configs/train.supervised.capsule_optimizer.toml`.

- `tests/`
  Local smoke tests for the capsule. Start with `tests/test_capsule_optimizer_smoke.py`.

- `resources/`
  Local contract/reference docs for agents and researchers.

## Contracts

Use `resources/LELABO_REFERENCE.md` for the exact signatures, return types,
context fields, and `DataBundle` structure.

Use `resources/PARAM_FLOW.md` for the exact param mapping from TOML to runtime.

Quick mental model:

- `models/` -> `@register_model(name)`
- `update_rules/` -> `@register_update_rule(name)`
- `datasets/` -> `@register_dataset(name)`
- `metrics/` -> `@register_metric(name, kind=...)`
- `initializers/` -> `@register_initializer(name)`
- `losses/` -> `@register_loss(name)`
- `optimizers/` -> `@register_optimizer(name)`
- `schedulers/` -> `@register_scheduler(name)`
- `callbacks/` -> `@register_callback(name)`

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
lelabo train supervised --config configs/train.supervised.quickstart.toml
```

## Common failure modes

- Plugin does not appear in `lelabo list`
  The `@register_*` line is still commented, or the file has an import error.

- `NameError: register_* is not defined`
  The decorator was uncommented but the matching import is missing.

- Trainer says a metric returned invalid data
  `compute()` / `finalize()` must return numeric scalars only.

- Local rule fails on a loss
  Some rules support only a subset of losses. Check the rule example and runtime error.

- Model works in BP but breaks for local rules
  Read `resources/MODEL_CACHE_ADVANCED.md` and `models/cache_walkthrough.py`.

## Mini prompt recipe for an agent

You can give an agent a prompt like:

> Open `AGENTS.md`, follow the capsule contracts, implement my new component in the right folder, keep only the decorator commented until the code is ready, then tell me which `lelabo list ...` and `pytest -q tests` commands to run.

That is usually enough for the agent to navigate the capsule correctly.
