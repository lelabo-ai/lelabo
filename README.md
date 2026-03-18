<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-mark-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-mark-light.svg">
  <img alt="LeLabo" src="docs/assets/logo-mark-light.svg" width="120">
</picture>

# LeLabo

**A modular research library for learning algorithms.**

[![version](https://img.shields.io/badge/version-0.1.0-7C5CFF?style=flat-square)](https://github.com/lelabo-ai/lelabo)
[![python](https://img.shields.io/badge/python-3.11_|_3.12-4B8BBE?style=flat-square)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](LICENSE)

[Documentation](https://lelabo-ai.github.io/) · [Quickstart](#quickstart) · [Install](#install)

</div>

---

LeLabo is built for the point where toy scripts stop scaling but full frameworks start getting in the way. The core idea: keep the training runtime clean, make methods easy to swap, and treat reproducibility as a first-class output.

> **Status** — LeLabo is research-grade software. The public extension points are intentional, but some internal surfaces can still move. The strongest surface today is `supervised + capsules`.

## Why LeLabo

Most research code is written to reproduce one specific result. The method, the training loop, the dataset loading, the logging — everything is wired together in a single script. It works for the original paper, but as soon as you want to run the same method on a different dataset, compare it to another approach, or just check whether the reported numbers hold, you end up rewriting half the pipeline.

LeLabo separates the method from the infrastructure around it:

- models, datasets, update rules, optimizers, losses, callbacks, and metrics are all named components that can be swapped from a config file
- the training runtime is shared — two methods compared under LeLabo run through the same loop, the same logging, the same seed handling
- a capsule lets you package a method (model + rule + config + tests) without forking the project

In practice, this is useful in three situations that come up constantly in research:

- **Exploring a paper** — you want to test whether a method works on your own setup, not just on the dataset the authors used. If the method is a LeLabo component, you change the config and run it.
- **Building on a baseline** — you need a fair reference point. Instead of reimplementing a baseline or trusting someone else's script, you use the same runtime for both your method and the baseline.
- **Reviewing a contribution** — you want to verify empirically, not just read the numbers. A method that lives in a capsule can be re-run, inspected, and compared in minutes.

None of this is automatic — implementing a new method still requires work. What LeLabo removes is the surrounding plumbing: the training loop, the logging, the artifact writing, the sweep infrastructure. You focus on the method itself.

## Features

| | |
|---|---|
| **Config-driven training** | TOML configs with CLI overrides. Every run writes a `resolved_config.yaml` so you know exactly what ran. |
| **Update rules** | `bp`, `dfa`, `fa`, `drtp`, `dni`, `scl`, `softhebb` — swap from the config. |
| **Built-in models** | `mlp`, `cnn`, `resnet18`, `bert`, `deephebb` |
| **Parameter sweeps** | YAML grid configs, parallel execution, multi-GPU round-robin, `--dry-run` preview. |
| **W&B integration** | Optional. Metrics, configs, and summaries logged automatically. Sweep runs grouped in the dashboard. |
| **Capsules** | Local workspaces for custom models, rules, optimizers, datasets. Pack, share, reinstall. |
| **Run artifacts** | `meta.json`, `seeds.json`, `metrics.jsonl`, `summary.json` — reproducible by default. |

## Install

```bash
git clone https://github.com/lelabo-ai/lelabo.git
cd LeLabo
pip install -e "."
```

Optional extras:

```bash
pip install -e ".[dev]"     # development tools
pip install -e ".[wandb]"   # Weights & Biases support
```

Requires Python 3.11 or 3.12. For GPU support, install the CUDA-enabled PyTorch build for your machine.

## Quickstart

```bash
# See what's available
lelabo list models
lelabo list update-rules
lelabo list datasets

# Run the smallest baseline (iris + mlp + bp, CPU, ~5 seconds)
lelabo train supervised --config configs/train/supervised.quickstart.toml

# Override without editing the config
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --epochs 50 --lr 0.0005 \
  --set model.params.hidden=256

# Persist a run directory
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --run-dir outputs/my_first_run
```

## Sweeps

```yaml
# sweep.yaml
name: lr_search
display_keys: [rule, lr, seed]

base:
  dataset: mnist
  model: mlp
  epochs: 30
  optimizer: adamw

grid:
  rule: [bp, dfa]
  lr: [1e-2, 1e-3, 1e-4]
  seed: [0, 1, 2]
```

```bash
lelabo sweep run --config sweep.yaml --dry-run        # preview
lelabo sweep run --config sweep.yaml --max-parallel 4  # run
```

With W&B enabled, all sweep runs are grouped automatically in the dashboard:

```bash
export WANDB_PROJECT=my-project
lelabo sweep run --config sweep.yaml --max-parallel 4
```

## Capsules

```bash
lelabo capsule init my_capsule
cd my_capsule
```

A capsule is a local workspace for custom research code — models, update rules, optimizers, datasets, configs, sweeps, and tests. It can be packed, shared, and reinstalled through the CLI.

## CLI

```
lelabo train supervised    Train a supervised model
lelabo train rl            Train an RL agent
lelabo sweep run           Run a parameter sweep
lelabo list                List registered components
lelabo capsule             Create, pack, install capsules
lelabo audit               Audit a run directory
```

## Documentation

| Section | |
|---|---|
| [Getting Started](https://lelabo-ai.github.io/getting-started/) | Install and first experiment |
| [Concepts](https://lelabo-ai.github.io/concepts/) | Runtime, capsules, cache contract, artifacts |
| [Guides](https://lelabo-ai.github.io/guides/) | Sweeps, W&B, creating components |
| [Tutorials](https://lelabo-ai.github.io/tutorials/) | Reproduce a paper, compare rules, implement DFA |
| [Reference](https://lelabo-ai.github.io/reference/) | CLI flags, config schema, public API |
