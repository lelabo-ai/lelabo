# Supervised

This page is the public source of truth for the supervised golden paths.

## Main positioning

The main entry story is:

1. standard BP works cleanly
2. capsules let you inject your own idea quickly
3. local rules are available when you need them

## Golden path 1: BP tabular

Official config:

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

Reference setup:

- `iris + mlp + bp`

This path proves that LeLabo can run a clean baseline without research-specific setup.

## Golden path 2: Capsule optimizer path

Official flow:

```bash
lelabo capsule init my_capsule
cd my_capsule
```

Enable `capsule_sgd` in `optimizers/example.py`, then run:

```bash
lelabo list optimizers
```

```bash
lelabo train supervised --config configs/train/supervised.capsule_optimizer.toml
```

Reference setup:

- `mnist + cnn + bp + capsule_sgd`

This path proves that you can extend LeLabo without forking the built-in package.

## Golden path 3: BP vision

Official config:

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml
```

Reference setup:

- `cifar10 + cnn + bp`

This is the main vision baseline.

## Golden path 4: BP NLP / GLUE

Official config:

```bash
lelabo train supervised --config configs/train/supervised.glue.toml
```

Reference setup:

- `glue/sst2 + bert + bp`

This path proves that the supervised stack also supports modern Hugging Face workflows.

## Golden path 5: Local-rule reference

Official config:

```bash
lelabo train supervised --config configs/train/supervised.local_rule.toml
```

Reference setup:

- `mnist + mlp + dfa`

This path shows the non-BP side of LeLabo without making it the first onboarding step.

## Run artifacts

If you set `runtime.run_dir` or pass `--run-dir`, a supervised run writes:

- `meta.json`
- `resolved_config.yaml`
- `seeds.json`
- `metrics.jsonl`
- `summary.json`

Checkpointing is optional and disabled by default:

- enable it with `runtime.save_checkpoints = true` or `--save-checkpoints`
- LeLabo then writes `checkpoints/last.pt`
- if a best monitored state is available, it also writes `checkpoints/best.pt`

## What is official vs secondary

Official in this phase:

- the five paths above
- clean CLI usage
- consistent config-driven runs
- capsule extension workflow

Secondary in this phase:

- RL
- launchers and sweeps
- broader experiment-management workflows
