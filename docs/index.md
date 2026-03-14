# LeLabo

LeLabo is a research-oriented framework for experimenting with learning algorithms.

The current polished path is:

- run a clean supervised baseline from the CLI
- switch to local rules when needed
- create a capsule when you want to inject your own research component

## What is first-class today

- supervised BP baselines
- supervised local-rule experimentation
- built-in models for tabular, vision, and GLUE/BERT workflows
- capsule scaffolding for custom extensions

## Recommended reading order

1. [Installation](installation.md)
2. [Quickstart](quickstart.md)
3. [Supervised](supervised.md)
4. [Capsules](capsules.md)
5. [Cache and local rules](cache-local-rules.md)

## Official supervised paths

- BP tabular: `iris + mlp + bp`
- Capsule path: `mnist + cnn + bp + capsule_sgd`
- BP vision: `cifar10 + cnn + bp`
- BP NLP: `glue/sst2 + bert + bp`
- Local-rule path: `mnist + mlp + dfa`

LeLabo is still optimized for research iteration, not API stability.
