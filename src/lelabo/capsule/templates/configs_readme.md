# Configs

This folder contains the capsule entry points you are expected to run first.

If you are onboarding fast, use them in this order:

1. `configs/train.supervised.quickstart.toml`
   `iris + mlp + bp`
2. `configs/train.supervised.capsule_optimizer.toml`
   `mnist + cnn + bp + capsule_sgd`
3. `configs/train.supervised.detailed.toml`
   richer supervised baseline
4. `configs/train.rl.detailed.toml`
   RL example config

For the capsule extension path, the official first run is:

```bash
lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml
```

Before that, uncomment `@register_optimizer("capsule_sgd")` in `optimizers/example.py`.

Useful validation commands:

```bash
pytest -q tests
```

```bash
lelabo list optimizers
```

```bash
lelabo train supervised --config configs/train.supervised.quickstart.toml
```

If you are using a coding agent, read `../AGENTS.md` first. It explains which
folder to edit and where to find the precise local references.

Local references:

- `../resources/LELABO_REFERENCE.md`
- `../resources/PARAM_FLOW.md`
- `../resources/MODEL_CACHE_ADVANCED.md`

## Config style

LeLabo configs use `name + params` blocks for extensible components:

- `dataset`
- `model`
- `initializer`
- `loss`
- `update_rule`
- `optimizer`
- `scheduler`
- `callbacks`
- `metrics`

Layered config resolution still applies:

1. library defaults
2. config file
3. scalar CLI overrides
4. nested `--set key=value` overrides

## Extension starters in this capsule

- `models/example.py`
- `update_rules/example.py`
- `datasets/example.py`
- `metrics/example.py`
- `initializers/example.py`
- `losses/example.py`
- `optimizers/example.py`
- `schedulers/example.py`
- `callbacks/example.py`

For advanced model cache patterns, see `models/cache_walkthrough.py`.
