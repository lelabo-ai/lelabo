# Capsule Extension Guide

This file is the hub for capsule work.

Use it to choose the right extension type, find the right folder, and open the
right local reference file. For normal capsule work, do not start by reading
LeLabo internals.

## Official first path

1. Uncomment `@register_optimizer("capsule_sgd")` in `optimizers/example.py`
2. Run `lelabo registries optimizers`
3. Run `lelabo train supervised --config configs/train/supervised.capsule_optimizer.toml`
4. Run `pytest -q tests`

## Open these files in this order

1. `README.md`
2. `AGENTS.md`
3. `resources/EXTENSION_RECIPES.md`
4. `resources/LELABO_REFERENCE.md`
5. `resources/PARAM_FLOW.md`
6. `resources/UPDATE_RULE_LIFECYCLE.md` only if you are implementing a custom update rule
7. `resources/MODEL_CACHE_ADVANCED.md` only if you need intermediate activations or block semantics
8. `resources/PAPER_PACK_PLAYBOOK.md` only if the work spans several components

Start from the matching `example.py` in the target folder and keep the
decorator commented until the component is ready.

## If your idea is X, edit Y

- New architecture or model: `models/`
- New learning or update logic: `update_rules/`
- New dataset loader or split logic: `datasets/`
- New train/eval metric: `metrics/`
- New parameter initializer: `initializers/`
- New loss: `losses/`
- New optimizer: `optimizers/`
- New scheduler: `schedulers/`
- New trainer callback: `callbacks/`
- Several moving parts: start with `resources/PAPER_PACK_PLAYBOOK.md`

## Decision Guide

- If config-only or parameter-only changes are enough, prefer config changes before custom code.
- If the architecture changes but the training logic does not, edit `models/`.
- If the training logic or credit assignment changes, edit `update_rules/`.
- If the main change is loading, splitting, or dataset metadata, edit `datasets/`.
- If the work spans several components, use the paper-pack path.
- Open cache docs only if the extension needs intermediate activations or block semantics.

## Which file answers which question?

- Where do I start? -> `AGENTS.md`
- How do I add X? -> `resources/EXTENSION_RECIPES.md`
- What is the exact signature? -> `resources/LELABO_REFERENCE.md`
- Where do my params arrive? -> `resources/PARAM_FLOW.md`
- Do I need cache/block semantics? -> `resources/MODEL_CACHE_ADVANCED.md`
- What may an update rule do? -> `resources/UPDATE_RULE_LIFECYCLE.md`
- Does my paper fit the runtime? -> `resources/PAPER_PACK_PLAYBOOK.md`

## Stop Signs

- Do not edit LeLabo internals for normal capsule extensions
- Do not add cache logic to a BP-only model unless a rule really needs it
- Do not hide dataset logic inside a model
- Do not start with one giant opaque script

## Validation commands

Run these from the capsule root:

```bash
pytest -q tests
```

```bash
lelabo registries optimizers
```

```bash
lelabo train supervised --config configs/train/supervised.capsule_optimizer.toml
```

Useful follow-up commands:

```bash
lelabo registries models
```

```bash
lelabo registries update-rules
```

```bash
lelabo train supervised --config configs/train/supervised.paper_pack.toml
```

## Common failure modes

- Plugin does not appear in `lelabo registries`
  The `@register_*` line is still commented, or the file has an import error.

- Params look missing at runtime
  Read `resources/PARAM_FLOW.md` and verify you are using the documented source.

- A rule needs intermediate activations or block semantics
  Read `resources/MODEL_CACHE_ADVANCED.md`.

- The extension spans several components and starts feeling ad hoc
  Read `resources/PAPER_PACK_PLAYBOOK.md`.
