# LeLabo

LeLabo is a research-oriented framework for experimenting with learning algorithms.

It is built around a few simple ideas:

- models are regular PyTorch `nn.Module`s
- training logic is selected through update rules such as `bp`, `dfa`, `drtp`, `fa`, `dni`, `scl`, and `softhebb`
- datasets, models, losses, optimizers, metrics, schedulers, and update rules are all registry-driven
- the cache provider can instrument models at runtime to support local learning rules

This project is currently optimized for research iteration, not API stability.

## Current scope

LeLabo currently provides:

- supervised training from the CLI
- RL training entrypoints
- built-in models for MLPs, CNNs, ResNets, and Hugging Face classifiers
- local-rule support through the standard cache provider
- capsule scaffolding for custom research extensions

## Typical workflow

1. Inspect what is available with `lelabo list`.
2. Launch a baseline BP run from the CLI.
3. Create a capsule when you want to inject your own optimizer, model, metric, or rule.
4. Move to CIFAR-10, GLUE, or local-rule configs once the baseline path is clear.

## Example

```bash
lelabo train supervised --dataset iris --model mlp --algo bp
```

Official supervised onboarding paths:

- BP tabular: `iris + mlp + bp`
- Capsule path: custom `capsule_sgd` on `mnist + cnn + bp`
- BP vision: `cifar10 + cnn + bp`
- BP NLP: `glue/sst2 + bert + bp`
- Local-rule path: `mnist + mlp + dfa`

## Documentation

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [API overview](api.md)
- [Research notes](research-notes.md)

Repository: [GitHub](https://github.com/adrienkegreisz/lelabo)
