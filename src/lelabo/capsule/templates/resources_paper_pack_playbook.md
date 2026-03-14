# Paper Pack Playbook

Use this file when you are implementing a full paper in a capsule.

The goal is to decompose the paper cleanly into LeLabo components instead of building one opaque script.

## What a paper pack is

A paper pack is usually some combination of:

- model
- update rule
- dataset
- config
- test

Sometimes it also needs:

- metric
- loss
- scheduler
- callback

## Implementation order

Recommended order:

1. choose the official closest baseline config
2. decide whether the paper really needs a custom dataset
3. implement the model if needed
4. implement the update rule if needed
5. add config(s)
6. add a smoke test

Do not start by editing everything at once.

## Where each concern should live

- architecture change -> `models/`
- update logic / local rule -> `update_rules/`
- split logic / loader logic -> `datasets/`
- experiment parameters -> `configs/`
- minimal regression coverage -> `tests/`

## If the paper uses local learning

Read:

- `resources/MODEL_CACHE_ADVANCED.md`
- `resources/UPDATE_RULE_LIFECYCLE.md`

## If the paper is mostly BP-compatible

Prefer the smallest possible change set:

- reuse built-in model if possible
- reuse built-in loss if possible
- reuse built-in optimizer if possible

Only add custom code where the paper really diverges.

## Runtime-fit check

Before you commit to an implementation shape, ask:

- does the paper fit a normal step-based rule?
- does it need intermediate activations?
- does it need several explicit phases?
- does it need several loaders or several optimizers as first-class runtime concepts?

If the answer is mostly “yes” to the last two, document the limitation clearly. The current runtime may still support the paper, but not as a perfect first-class staged program.

## Official mini paper pack in this scaffold

This capsule ships a small multi-component reference path:

- model: `example_mlp`
- update rule: `local_head`
- config: `configs/train.supervised.paper_pack.toml`
- smoke test: `tests/test_paper_pack_smoke.py`

This is not a scientific reference implementation. It is an integration reference.

## Validation checklist

Before calling the paper pack “done”, verify:

- the registered plugins appear in `lelabo list ...`
- the config runs
- the smoke test passes
- the runtime contract is documented if the paper needs non-standard assumptions

## Done criteria

The paper pack is in good shape when:

- each concern lives in the right folder
- the config is readable
- the smoke test covers the intended integration path
- an agent can navigate the capsule without reading LeLabo internals
