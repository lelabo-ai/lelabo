# LeLabo

<p class="lead"><i>
Implement the method, not the whole lab around it.
</i>
</p>

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Learn the concepts](concepts/index.md){ .md-button }

---

Documentation: [https://adrienkegreisz.github.io/lelabo-doc/](https://github.com/adrienkegreisz/LeLabo)

Source code: [https://github.com/adrienkegreisz/LeLabo/](https://github.com/adrienkegreisz/LeLabo)

---

It provides a mature supervised runtime, structured run artifacts, and a capsule-based extension workflow so methods can be integrated, reused, compared, and stress-tested more easily across projects.

## Why LeLabo exists

Research code is often tied to a single paper setup:
- one dataset
- one model family
- one training script
- one evaluation path
- one opaque repository

That makes methods harder to reuse, compare fairly, or integrate into another workflow.

LeLabo is built to reduce that friction.

Instead of rebuilding the same experimental plumbing for every project, you can focus on the method itself while relying on shared runtime contracts, structured outputs, and reusable extension points.

## What you can do today

Today, the most mature public surface of LeLabo is:

- supervised experimentation
- capsule-based extension workflows
- cache and local-rule public contracts
- structured run artifacts
- CLI workflows outside RL

Reinforcement learning support is under active development and is not yet the main mature public surface.

## Core ideas

### Supervised runtime

LeLabo provides a supervised training runtime centered on a clear mental model:
- `Trainer` orchestrates training and evaluation
- `learner` owns update logic through `train_step(...)`
- `loss`, `metrics`, `callbacks`, and `schedulers` follow explicit runtime contracts
- outputs are structured as `FitResult`, `EpochRecord`, `SplitSummary`, and run artifacts

### Capsules

A capsule is a local extension workspace for a method, paper implementation, or research project.

Capsules are the main way to extend LeLabo without starting from framework internals. They are designed to make custom research code easier to structure, rerun, package, and share.

### Reusable research components

LeLabo is built toward a workflow where methods are easier to:
- integrate into new projects
- reuse as baselines
- compare under different datasets or settings
- inspect beyond a single paper repository

## Start here

If you are new to LeLabo, follow this path:

1. [Install LeLabo](getting-started/installation.md)
2. [Run the quickstart](getting-started/quickstart.md)
3. [Understand the supervised runtime](concepts/supervised-runtime.md)
4. [Learn how capsules work](concepts/capsules.md)

## Documentation map

- **Getting Started**: install LeLabo and run your first experiment
- **Concepts**: understand the runtime, capsules, cache/local rules, and run artifacts
- **Guides**: create models, update rules, datasets, capsules, and paper packs
- **Reference**: exact CLI and public runtime contracts
- **Research Notes**: limitations, design choices, and future direction
- **Community**: contributing and sharing capsules

## Community and sharing

LeLabo is meant to grow as a shared research ecosystem.

The long-term goal is to make methods easier to package as reusable capsules, so researchers can share implementations, integrate baselines more easily, and test methods beyond their original paper setup.

Contributions are welcome across runtime, capsules, documentation, and research integrations.
