# LeLabo

LeLabo is a modular research library for learning algorithms.

It is built for the point where toy scripts stop scaling, but full framework work starts getting in the way. The core idea is simple: keep the training runtime clean, make methods easy to swap, and treat reproducibility as a first-class output.

What is solid today:

- config-driven supervised training from the CLI
- alternative credit-assignment and local-learning rules
- structured run artifacts for reproducibility
- capsule-based extensions for custom research code
- parameter sweeps and optional Weights & Biases tracking

LeLabo is still research-grade software. The public extension points are intentional, but some internal surfaces can still move.

## Why LeLabo

Most research code is written to reproduce one specific result. The method, the training loop, the dataset loading, the logging — everything is wired together in a single script. It works for the original paper, but as soon as you want to run the same method on a different dataset, compare it to another approach, or just check whether the reported numbers hold, you end up rewriting half the pipeline.

LeLabo separates the method from the infrastructure around it:

- models, datasets, update rules, optimizers, losses, callbacks, and metrics are all named components that can be swapped from a config file
- the training runtime is shared — two methods compared under LeLabo run through the same loop, the same logging, the same seed handling
- a capsule lets you package a method (model + rule + config + tests) without forking the project

In practice, this is useful in three situations that come up constantly in research:

- **exploring a paper** — you want to test whether a method works on your own setup, not just on the dataset the authors used. If the method is a LeLabo component, you change the config and run it.
- **building on a baseline** — you need a fair reference point. Instead of reimplementing a baseline or trusting someone else's script, you use the same runtime for both your method and the baseline.
- **reviewing or evaluating a contribution** — you want to verify empirically, not just read the numbers. A method that lives in a capsule can be re-run, inspected, and compared in minutes.

None of this is automatic — implementing a new method still requires work. What LeLabo removes is the surrounding plumbing: the training loop, the logging, the artifact writing, the sweep infrastructure. You focus on the method itself.

## What You Can Do

- run a clean baseline in seconds with `iris + mlp + bp`
- switch between rules such as `bp`, `dfa`, `fa`, `drtp`, `dni`, `scl`, and `softhebb`
- use built-in models such as `mlp`, `cnn`, `resnet18`, `bert`, and `deephebb`
- launch grid sweeps from YAML configs
- track runs locally or send them to W&B
- scaffold a capsule and add your own optimizer, model, dataset, metric, scheduler, callback, or update rule

## Install

From source:

```bash
git clone https://github.com/adrienkegreisz/LeLabo.git
cd LeLabo
pip install -e "."
```

Development tools:

```bash
pip install -e ".[dev]"
```

Optional W&B support:

```bash
pip install -e ".[wandb]"
```

Requirements:

- Python 3.11 or 3.12
- `pip`

If you need GPU support, install the CUDA-enabled PyTorch build appropriate for your machine.

## Quickstart

Inspect the built-in registries:

```bash
lelabo list models
lelabo list update-rules
lelabo list datasets
```

Run the smallest official baseline:

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

That launches `iris + mlp + bp`, which is the recommended first run because it is fast, CPU-friendly, and proves that the full stack is wired correctly.

Override from the CLI without editing the config:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --epochs 50 \
  --lr 0.0005 \
  --set model.params.hidden=256
```

Persist a run directory:

```bash
lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --run-dir outputs/my_first_run
```

LeLabo then writes:

- `meta.json`
- `resolved_config.yaml`
- `seeds.json`
- `metrics.jsonl`
- `summary.json`
- optional `checkpoints/` with `--save-checkpoints`

## Official Supervised Paths

The current first-class supervised paths are:

1. `iris + mlp + bp`
2. `mnist + cnn + bp + capsule_sgd`
3. `cifar10 + cnn + bp`
4. `glue/sst2 + bert + bp`
5. `mnist + mlp + dfa`

Reference configs live in `configs/train/`.

## Sweeps

LeLabo includes a sweep runner that expands a YAML grid into multiple `lelabo train` jobs.

```bash
lelabo sweep run --config experiments/sweeps/demo.yaml
```

Useful flags:

- `--dry-run` to preview generated commands
- `--max-parallel N` to run several jobs at once
- `--gpus 0,1,2` for round-robin GPU assignment

Sweep outputs are written under `outputs/runs/<sweep_name>/`, with one folder per run plus a `plan.json` for the full sweep plan.

## Weights & Biases

W&B support is optional. If `wandb` is installed and a project is configured, LeLabo logs run metrics and groups sweep jobs automatically.

```bash
wandb login
export WANDB_PROJECT=my-project

lelabo train supervised \
  --config configs/train/supervised.quickstart.toml \
  --run-dir outputs/run_001
```

You can also configure it directly in TOML:

```toml
[wandb]
project = "my-project"
entity = "my-team"
tags = ["baseline", "iris"]
group = "quick-tests"
enabled = true
```

Without W&B, everything still works locally from the run artifacts on disk.

## Capsules

A capsule is a local workspace for custom research extensions. It lets you add new components without editing the core package.

Create one with:

```bash
lelabo capsule init my_capsule
cd my_capsule
```

Capsules can hold configs, sweeps, custom builders, tests, and reusable method implementations. They can also be packed, installed, stashed, and restored through the CLI.

## CLI Surface

```bash
lelabo --help
```

Main commands:

- `lelabo train supervised`
- `lelabo train rl`
- `lelabo sweep run`
- `lelabo list`
- `lelabo capsule`
- `lelabo audit`

## Documentation

- [Docs Home](docs/index.md)
- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [Configuration](docs/concepts/configuration.md)
- [Supervised Runtime](docs/concepts/supervised-runtime.md)
- [Run Artifacts](docs/concepts/run-artifacts.md)
- [Run Supervised Experiments](docs/guides/run-supervised.md)
- [Run Parameter Sweeps](docs/guides/sweeps.md)
- [Weights & Biases Integration](docs/guides/wandb.md)
- [Create a Capsule](docs/guides/create-capsule.md)
- [CLI Reference](docs/reference/cli.md)

## Status

The strongest surface today is `supervised + capsules`.

RL support exists, but it is not yet the primary stable surface. Sweeps and experiment-management workflows are available and usable, but the most mature documentation and conventions still center on the supervised runtime and capsule workflow.
