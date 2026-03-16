# Create a Capsule

A capsule is your workspace for custom research code. This guide walks through creating one and running the first official extension path.

## Create the scaffold

```bash
lelabo capsule init my_capsule
cd my_capsule
```

LeLabo creates a structured directory with example files for every extension type, local documentation, and ready-to-run configs.

## Run the first extension path

The official first path is the optimizer extension. It is the simplest possible capsule workflow — one file, one decorator, one run.

**Step 1.** Open `optimizers/example.py` and uncomment the `@register_optimizer` decorator:

```python
from lelabo.optimizers import OptimizerContext, register_optimizer
import torch

@register_optimizer("capsule_sgd")  # ← uncomment this line
def build_capsule_sgd(ctx: OptimizerContext):
    return torch.optim.SGD(
        ctx.param_groups,
        lr=ctx.lr,
        momentum=ctx.optimizer_params().get("momentum", 0.9),
    )
```

**Step 2.** Verify the component appears in the registry:

```bash
lelabo list optimizers
# capsule_sgd should appear in the list
```

**Step 3.** Run the capsule optimizer config:

```bash
lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml
```

That runs `mnist + cnn + bp + capsule_sgd`. Your custom optimizer is now part of the runtime — no core code touched.

## What's inside the capsule

```
my_capsule/
├── README.md                        ← start here
├── AGENTS.md                        ← for agent-oriented workflows
├── capsule.toml                     ← capsule metadata
├── models/example.py
├── update_rules/example.py
├── datasets/example.py
├── optimizers/example.py            ← the first extension path
├── losses/example.py
├── metrics/example.py
├── schedulers/example.py
├── callbacks/example.py
├── configs/
│   ├── train.supervised.quickstart.toml
│   ├── train.supervised.capsule_optimizer.toml
│   └── train.supervised.paper_pack.toml
├── tests/
└── resources/
    ├── EXTENSION_RECIPES.md         ← step-by-step recipes
    ├── LELABO_REFERENCE.md          ← exact contracts
    ├── PARAM_FLOW.md                ← how params flow from config to builder
    └── UPDATE_RULE_LIFECYCLE.md
```

!!! tip "Local docs are the source of truth for extensions"
    `resources/EXTENSION_RECIPES.md` inside the capsule has precise, step-by-step recipes for every extension type. Use those when implementing a method — they are more detailed than this public page.

## Run the capsule tests

```bash
pytest -q tests/
```

The scaffold includes smoke tests for the capsule optimizer path. Add your own tests as you add components.

## Stash and share a capsule

When you're done working, stash the capsule into the local store:

```bash
lelabo capsule stash
```

To build a shareable bundle:

```bash
lelabo capsule pack --from outputs/my_run
```

To install a bundle someone shared with you:

```bash
lelabo capsule install path/to/bundle.tar.gz
```

## Next steps

- [Create a model](create-model.md) — register a custom architecture
- [Create an update rule](create-update-rule.md) — register a custom learning rule
- [Create a dataset](create-dataset.md) — register a custom data loader
- [Build a paper pack](paper-pack.md) — package a full method implementation
