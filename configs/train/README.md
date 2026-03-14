# Train Configs

This folder contains the official config entrypoints for `lelabo train`.

For the public supervised story, this page is the operator-facing companion to `docs/supervised.md`.

## Official supervised configs

### `configs/train/supervised.quickstart.toml`

- dataset: `iris`
- model: `mlp`
- update rule: `bp`
- role: first BP tabular baseline

### Capsule path

Official flow:

1. `lelabo create capsule my_capsule`
2. enable `capsule_sgd` in `optimizers/example.py`
3. run `configs/train.supervised.capsule_optimizer.toml` from the capsule root

Role:

- first extension path
- proves optimizer injection without forking the built-in package

### `configs/train/supervised.detailed.toml`

- dataset: `cifar10`
- model: `cnn`
- update rule: `bp`
- role: main BP vision baseline

### `configs/train/supervised.glue.toml`

- dataset: `glue/sst2`
- model: `bert`
- update rule: `bp`
- role: main BP NLP baseline

You can also switch this config to `stsb` for regression-oriented validation.

### `configs/train/supervised.local_rule.toml`

- dataset: `mnist`
- model: `mlp`
- update rule: `dfa`
- role: main local-rule reference path

## Resolution order

Runtime precedence is:

1. library defaults
2. project TOML
3. explicit CLI flags
4. `--set key=value`

`--config` is optional. If present, it overrides auto-detected project config paths.

## Examples

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

```bash
lelabo train supervised --config configs/train/supervised.detailed.toml --set optimizer.params.lr=0.02
```

```bash
lelabo train supervised --config configs/train/supervised.glue.toml
```

```bash
lelabo train supervised --config configs/train/supervised.local_rule.toml
```

## Run artifacts

When `runtime.run_dir` is set, LeLabo writes:

- `meta.json`
- `resolved_config.yaml`
- `seeds.json`
- `metrics.jsonl`
- `summary.json`

Checkpointing is opt-in and disabled by default:

- set `runtime.save_checkpoints = true`
- or pass `--save-checkpoints`
- checkpoints are written under `checkpoints/`

## Official examples vs local variants

Official examples:

- are part of the documented golden paths
- are expected to stay readable and stable
- are the configs new users should discover first

Local variants:

- are normal and encouraged
- should not replace the official examples as the main onboarding path

## Capsule references

Inside a generated capsule, read:

- `AGENTS.md`
- `configs/README.md`
- `resources/LELABO_REFERENCE.md`
- `resources/PARAM_FLOW.md`
- `resources/MODEL_CACHE_ADVANCED.md`

## RL note

`configs/train/rl.detailed.toml` remains available, but RL is not part of the main polished documentation scope in this phase.
