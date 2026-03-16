---
hide:
  - toc
  - navigation
---

<div class="lelabo-hero" markdown>

<div class="lelabo-hero__title-row">
<img src="assets/logo-mark.svg" class="lelabo-hero__mark" alt="">
<span class="lelabo-hero__name">LeLabo</span>
</div>

<p class="lelabo-hero__tagline">A modular research library for learning algorithms.</p>

[![version](https://img.shields.io/badge/version-0.1.0-7C5CFF?style=flat-square)](https://github.com/adrienkegreisz/LeLabo) [![python](https://img.shields.io/badge/python-3.11_|_3.12-4B8BBE?style=flat-square)](https://www.python.org) [![downloads](https://img.shields.io/badge/downloads-1.2k%2Fmonth-2ea44f?style=flat-square)](https://github.com/adrienkegreisz/LeLabo) [![license](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](https://github.com/adrienkegreisz/LeLabo/blob/main/LICENSE)
{: .lelabo-hero__badges }

[Get started](getting-started/installation.md){ .md-button .lelabo-btn-primary }
[Explore concepts](concepts/index.md){ .md-button .lelabo-btn-secondary }

</div>

---

## Why LeLabo exists

Research code is often tied to a single paper setup: one dataset, one model family, one training script, one opaque repository.

That makes methods harder to reuse, compare fairly, or integrate into another workflow.

LeLabo is built to reduce that friction. Instead of rebuilding the same experimental infrastructure for every project, you focus on the method itself — the runtime, the extension points, and the reproducibility are already there.

---

## What is stable today

<div class="grid cards" markdown>

-   **Supervised runtime**

    ---

    A clean training loop with explicit control flow. `Trainer` orchestrates the model, learner, loss, callbacks, metrics, and schedulers. Outputs are structured and reproducible.

    [:octicons-arrow-right-24: Supervised runtime](concepts/supervised-runtime.md)

-   **Capsule-based extensions**

    ---

    A capsule is a local workspace for your research code. Add a custom model, update rule, optimizer, or dataset without touching the core package.

    [:octicons-arrow-right-24: What are capsules?](concepts/capsules.md)

-   **Cache & local rules**

    ---

    Local learning rules need intermediate activations. LeLabo provides a stable public contract — `declare_blocks()`, `CacheSpec`, `forward_with_standard_cache()` — without requiring a custom training loop.

    [:octicons-arrow-right-24: Cache & local rules](concepts/cache-local-rules.md)

-   **Structured run artifacts**

    ---

    Every run can write `meta.json`, `resolved_config.yaml`, `seeds.json`, `metrics.jsonl`, and `summary.json`. Reproducible by default.

    [:octicons-arrow-right-24: Run artifacts](concepts/run-artifacts.md)

</div>

---

## A first run

```bash
pip install lelabo
```

```bash
lelabo train supervised --config configs/train/supervised.quickstart.toml
```

That runs `iris + mlp + bp` — the smallest official baseline. From there, every component is swappable from the config.

```bash
lelabo list models
lelabo list update-rules
lelabo list datasets
```

---

## The longer ambition

LeLabo is designed toward a workflow where methods travel beyond their original paper setup.

A capsule can hold a full method implementation — model, update rule, config, tests. It can be stashed, packed, shared, and reinstalled. The goal is to make it easier to:

- reuse a baseline without rebuilding it
- compare methods under the same runtime conditions
- audit a result outside its original repository

That is still a direction, not a promise. The supervised runtime and capsule workflow are stable today. The ecosystem around sharing and reuse is being built.

---

## Reinforcement learning

RL support exists but is not the primary stable surface yet. See the [RL page](rl.md) for the current status.

---

## Documentation map

| Section | What you will find |
|---|---|
| [Getting Started](getting-started/index.md) | Install LeLabo and run your first experiment |
| [Concepts](concepts/index.md) | The runtime model, capsules, cache contract, run artifacts |
| [Guides](guides/index.md) | How to create models, update rules, datasets, capsules |
| [Reference](reference/index.md) | Exact CLI flags, config schema, public API |
| [Research Notes](research-notes.md) | Design choices, limitations, future direction |
| [Community](community.md) | Contributing, sharing capsules |
