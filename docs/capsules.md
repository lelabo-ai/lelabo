# Capsules

Capsules are the local extension workspace for LeLabo.

Use a capsule when you want to add your own research component without editing the main package.

## What belongs in a capsule

- custom models
- update rules
- datasets
- metrics
- losses
- optimizers
- schedulers
- callbacks

## Official first capsule path

The official first extension path is the optimizer path:

```bash
lelabo capsule init my_capsule
cd my_capsule
```

Then:

1. open `optimizers/example.py`
2. uncomment `@register_optimizer("capsule_sgd")`
3. run `lelabo list optimizers`
4. run `lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml`

## Why capsules matter

Capsules are the bridge between:

- the public CLI-first workflow
- your own research code
- agent-oriented local extension docs

They are designed so that a researcher can hand the capsule to an agent, describe the desired extension, and have the agent work from the local contracts instead of reverse-engineering the main repo.

## Public docs vs local capsule docs

This public page explains what a capsule is and when to use it.

The detailed agent-oriented extension contracts live inside the generated capsule:

- `README.md`
- `AGENTS.md`
- `resources/LELABO_REFERENCE.md`
- `resources/PARAM_FLOW.md`
- `resources/MODEL_CACHE_ADVANCED.md`

Additional advanced resources can be added there without turning the public site into a full internal contract manual.

## When to create a capsule

Create one when:

- built-in components are close, but not enough
- you want a custom optimizer or scheduler quickly
- you want to prototype a new model or update rule
- you want a self-contained extension workspace you can share or archive

Do not create one just to run the built-in golden paths.
