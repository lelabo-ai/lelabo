# Capsules

A capsule is a local extension workspace for LeLabo. It is where your research code lives — your models, update rules, datasets, optimizers, or any other custom component.

## What a capsule is

When you run `lelabo capsule init my_capsule`, LeLabo creates a structured directory:

```
my_capsule/
├── README.md
├── AGENTS.md
├── capsule.toml
├── models/
│   └── example.py
├── update_rules/
│   └── example.py
├── datasets/
│   └── example.py
├── optimizers/
│   └── example.py
├── losses/
│   └── example.py
├── metrics/
│   └── example.py
├── schedulers/
│   └── example.py
├── callbacks/
│   └── example.py
├── configs/
│   └── train/supervised.quickstart.toml
├── tests/
└── resources/
    ├── EXTENSION_RECIPES.md
    ├── LELABO_REFERENCE.md
    └── ...
```

Each subfolder corresponds to a registry. Any component you register in a capsule becomes available to the CLI by name — without modifying the core package.

## Why capsules exist

The alternative to capsules is forking LeLabo or maintaining a separate training script for each project. Both approaches make it harder to compare methods fairly, reuse code across projects, or share a working implementation with someone else.

A capsule gives you a self-contained workspace that:

- plugs into the existing runtime without patching it
- can be stashed, packed, versioned, and reinstalled
- documents its own extension contracts locally

## The official first path

The simplest way to see capsules work is the optimizer path:

```bash
lelabo capsule init my_capsule
cd my_capsule
```

Open `optimizers/example.py` and uncomment the `@register_optimizer` decorator:

```python
from lelabo.optimizers import OptimizerContext, register_optimizer

@register_optimizer("capsule_sgd")
def build_capsule_sgd(ctx: OptimizerContext):
    return torch.optim.SGD(
        ctx.param_groups,
        lr=ctx.lr,
        momentum=ctx.optimizer_params().get("momentum", 0.9),
    )
```

Then verify and run:

```bash
lelabo list optimizers       # capsule_sgd should appear
lelabo train supervised --config configs/train/supervised.capsule_optimizer.toml
```

Your custom optimizer is now part of the runtime — no core code touched.

## What you can put in a capsule

| Folder | Registry | Decorator |
|---|---|---|
| `models/` | Model registry | `@register_model` |
| `update_rules/` | Update rule registry | `@register_update_rule` |
| `datasets/` | Dataset registry | `@register_dataset` |
| `optimizers/` | Optimizer registry | `@register_optimizer` |
| `losses/` | Loss registry | `@register_loss` |
| `metrics/` | Metric registry | `@register_metric` |
| `schedulers/` | Scheduler registry | `@register_scheduler` |
| `callbacks/` | Callback registry | `@register_callback` |

## Capsule lifecycle

A capsule moves through a simple lifecycle:

```
init → (develop) → stash → push → install
                 ↘ pack  ↗
                 ↘ attach (link in-place)
```

| Command | What it does |
|---|---|
| `lelabo capsule init <name>` | Create a new capsule scaffold in the current directory |
| `lelabo capsule stash` | Move the capsule into the local capsule store |
| `lelabo capsule attach` | Link an existing local capsule into the store without moving files |
| `lelabo capsule checkout <id>` | Restore a stored capsule back into a workspace |
| `lelabo capsule pack --from <run-dir>` | Build a shareable `.tar.gz` bundle |
| `lelabo capsule install <bundle>` | Import a bundle or GitHub repo into the local store |
| `lelabo push` | Publish a capsule to GitHub (see the [push guide](../guides/push-capsule.md)) |
| `lelabo capsule list` | List stored capsules |
| `lelabo capsule remove <id>` | Remove a capsule from the store |

## Capsules as paper packs

A capsule can hold a full method implementation: model + update rule + configs + tests. This is called a paper pack.

The goal is to make it possible to take a method from a paper, implement it in a capsule, and then hand that capsule to someone else — who can run it, extend it, or compare it against their own baseline without rebuilding anything.

That direction is still being built, but the capsule mechanism is already the right primitive for it.

## Local docs vs public docs

This page explains what capsules are and how to use them.

The detailed extension contracts live inside the generated capsule itself:

- `README.md` — start here
- `AGENTS.md` — documentation hub designed for AI coding agents (Claude Code, Cursor, Codex, etc.). When an AI assistant opens your capsule, it reads this file automatically to understand the contracts and extension points. Human developers can ignore it and use `README.md` instead.
- `resources/EXTENSION_RECIPES.md` — step-by-step recipes for each extension type
- `resources/LELABO_REFERENCE.md` — exact signatures and contracts
- `resources/PARAM_FLOW.md` — how parameters flow from config to builder

If you are implementing a method, start from those local docs — they are more precise than this page.
