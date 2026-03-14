# LeLabo

LeLabo is a research-oriented framework for experimenting with learning algorithms.

The current polished surface is:

- supervised training from the CLI
- local-learning and alternative credit-assignment rules
- capsule scaffolding for custom research extensions

The project is still research-grade. Public extension points are intentional, but internals can move.

## What LeLabo is good at today

- running a clean BP baseline quickly
- switching between update rules such as `bp`, `dfa`, `drtp`, `fa`, `dni`, `scl`, and `softhebb`
- working with built-in models such as `mlp`, `cnn`, `resnet18`, `bert`, and `deephebb`
- creating a capsule to add your own optimizer, model, metric, dataset, scheduler, callback, or update rule

## Official supervised paths

These are the supervised paths currently treated as first-class:

1. `iris + mlp + bp`
2. capsule optimizer path: `mnist + cnn + bp + capsule_sgd`
3. `cifar10 + cnn + bp`
4. `glue/sst2 + bert + bp`
5. `mnist + mlp + dfa`

## Install

Minimal editable install:

```bash
pip install -e .
```

Supervised workflows:

```bash
pip install -e ".[supervised]"
```

NLP / GLUE workflows:

```bash
pip install -e ".[supervised,nlp]"
```

Local development:

```bash
pip install -e ".[supervised,rl,nlp,dev,test]"
```

## First commands

Inspect what is available:

```bash
lelabo list
```

Run the first BP baseline:

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

Run the main vision baseline:

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml
```

Create a capsule:

```bash
lelabo create capsule my_capsule
cd my_capsule
```

## Documentation

- [Home](docs/index.md)
- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Supervised](docs/supervised.md)
- [Capsules](docs/capsules.md)
- [Cache and local rules](docs/cache-local-rules.md)
- [API overview](docs/api.md)
- [Research notes](docs/research-notes.md)

## Scope and limitations

This phase of the documentation focuses on `supervised + capsules`.

RL, experiments, launchers, and tools remain available, but they are not documented at the same level of maturity yet.

The cache/local-rule runtime is strong for standard supervised and many local-rule setups, but staged multi-phase programs are not yet first-class runtime objects.
