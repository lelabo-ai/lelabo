# Reinforcement Learning

!!! warning "Under construction"
    RL support in LeLabo is not yet the primary stable surface. The documentation on this page reflects the current state honestly.

## What exists

LeLabo includes an RL stack:

- A set of built-in algorithms (`ppo` and others)
- An RL runner that integrates with the same `RunLogger` and artifact system as supervised training
- A `lelabo train rl` CLI command
- TOML config support for RL experiments

```bash
lelabo train rl \
  --env CartPole-v1 \
  --algo ppo \
  --run-dir outputs/rl_run
```

## What is not stable yet

- The RL surface is less tested and less documented than the supervised stack
- The API may change before it reaches the same maturity level
- Not all features available in `train supervised` are available in `train rl`

## When to use it

Use the RL stack if you need it and are comfortable with early-stage interfaces.

For anything production-grade or for reproducibility-critical experiments, the supervised stack is the right choice today.

## Roadmap

RL is actively being developed. The goal is to bring it to the same level of maturity as the supervised stack — stable contracts, full artifact support, and first-class documentation.

Follow the [GitHub repository](https://github.com/adrienkegreisz/LeLabo) for updates.
