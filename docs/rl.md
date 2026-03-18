# Reinforcement Learning

RL support in LeLabo is under active development. The supervised runtime, capsule system, and sweep infrastructure are the current stable surfaces — RL is next.

## What exists today

The codebase includes RL algorithms, runners, and configs. Some algorithms works, but they have not been through the same level of testing, documentation, and public review as the supervised stack.

Current RL components:

- basic policy-gradient and Q-learning algorithms
- a separate `lelabo train rl` entry point
- config-driven setup following the same patterns as supervised

## What is missing

- **Documentation** — no guides, no tutorials, no walkthrough yet.
- **Capsule integration** — RL components are not yet registered through the same capsule workflow as supervised components.
- **Sweep support** — sweeps work in principle (`lelabo sweep run` calls `lelabo train`), but RL-specific sweep patterns are not documented or tested.
- **Stability guarantees** — the RL API surface may change. Extension points that are stable for supervised are not necessarily stable for RL yet.

## Using it anyway

If you want to experiment with RL today:

```bash
lelabo train rl --config configs/train/rl.toml
```

## Timeline

There is no fixed release date. The plan is to bring RL to the same level as supervised once the current capsule and sweep surfaces are solid. Progress will be visible in the repository.
