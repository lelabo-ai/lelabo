# Guides

Step-by-step instructions for common tasks. Each guide assumes you have LeLabo installed and have read the relevant [Concepts](../concepts/index.md).

## Running experiments

| | |
|---|---|
| [Run supervised experiments](run-supervised.md) | Config-driven training, overrides, and reproducibility |
| [Run parameter sweeps](sweeps.md) | Grid search across hyperparameters with parallel execution |
| [Weights & Biases integration](wandb.md) | Experiment tracking, metric visualization, and run comparison |

## Extending LeLabo

All extensions live in a capsule. [Create a capsule](create-capsule.md) first if you haven't already.

| | |
|---|---|
| [Create a capsule](create-capsule.md) | Set up a capsule workspace and run the first extension path |
| [Create a model](create-model.md) | Register a custom `nn.Module` |
| [Create an update rule](create-update-rule.md) | Register a custom learning rule |
| [Create a dataset](create-dataset.md) | Register a custom data loader |

## Sharing & Reuse

| | |
|---|---|
| [Push a capsule](push-capsule.md) | Publish a capsule to GitHub and install it on another machine |
| [Build a paper pack](paper-pack.md) | Package a full method implementation as a shareable capsule |
