# Reproduce a Paper in a Capsule

This tutorial walks through the most common LeLabo workflow: taking a published method, implementing it in a capsule, running experiments, and packaging the result.

We'll use the **Muon optimizer** as our example — a momentum-based optimizer with orthogonalization that works well on small-to-medium models. The method changes only the optimizer, not the model or the training rule, which makes it a clean first case.

## The approach

Before writing any code, identify what the paper actually changes relative to a standard setup:

| Component | Standard | Paper |
|---|---|---|
| Model | MLP / CNN | **Same** — no architecture change |
| Update rule | BP | **Same** — standard backpropagation |
| Optimizer | AdamW | **Custom** — Muon optimizer |
| Dataset | MNIST / CIFAR-10 | **Same** |

Only the optimizer changes. Everything else stays builtin.

!!! tip "Use builtins whenever possible"
    LeLabo ships with well-tested models, datasets, and rules. If a builtin covers your need, use it. This keeps experiments comparable across the community and avoids duplicating tested code.

    If a builtin is close but not quite right, [open an issue](https://github.com/adrienkegreisz/LeLabo/issues) — it's better to improve the shared component than to reimplement it locally.

## Step 1 — Create the capsule

```bash
lelabo capsule init muon_paper
cd muon_paper
```

## Step 2 — Implement the optimizer

Open `optimizers/example.py` and replace the example with the Muon optimizer:

```python
import torch
from torch import Tensor
from lelabo.optimizers import OptimizerContext, register_optimizer


def _zeropower_via_newtonschulz(G: Tensor, steps: int = 5) -> Tensor:
    """Approximate the orthogonal component of a matrix via Newton-Schulz iteration."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X


@register_optimizer("muon")
def build_muon(ctx: OptimizerContext) -> torch.optim.Optimizer:
    params = ctx.optimizer_params()
    lr = ctx.lr
    momentum = params.get("momentum", 0.95)
    nesterov = params.get("nesterov", True)
    ns_steps = params.get("ns_steps", 5)

    class Muon(torch.optim.Optimizer):
        def __init__(self, params, lr, momentum, nesterov, ns_steps):
            defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad

                    if g.ndim >= 2:
                        g = _zeropower_via_newtonschulz(g, steps=ns_steps)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        update = g + momentum * buf
                    else:
                        update = buf

                    p.add_(update, alpha=-lr)

    return Muon(ctx.params, lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
```

## Step 3 — Verify registration

```bash
lelabo list optimizers
# muon should appear in the list
```

## Step 4 — Configure the experiment

Edit `configs/train/supervised.paper_pack.toml`:

```toml
config_version = "auto"
lelabo_version = "auto"
task = "supervised"

[dataset]
name = "mnist"

[model]
name = "mlp"
[model.params]
hidden = 256
layers = 3
activation = "relu"

[loss]
name = "ce"

[update_rule]
name = "bp"

[optimizer]
name = "muon"
[optimizer.params]
lr = 0.02
momentum = 0.95
nesterov = true
ns_steps = 5

[runtime]
device = "auto"
seed = 42
determinism = "relaxed"
display = "compact"
run_dir = "outputs/muon_mnist"

[train]
epochs = 20
batch = 64

[[callbacks]]
name = "earlystopping"
enabled = true
[callbacks.params]
monitor = "val.acc"
patience = 5
restore_best = true

[[metrics]]
name = "acc"
```

Notice: `model`, `dataset`, `loss`, `update_rule` are all builtins. Only `optimizer` is custom.

## Step 5 — Quick sanity check

Run once to make sure the optimizer works:

```bash
lelabo train supervised --config configs/train/supervised.paper_pack.toml
```

Check the output — if training converges and accuracy is reasonable, the implementation is wired correctly. This is not the real experiment yet, just a smoke test.

## Step 6 — Run the real comparison as a sweep

A single run on a single seed does not tell you much. To actually compare Muon against AdamW, you need multiple seeds and both optimizers under the same conditions.

Create a sweep config in `sweeps/muon_vs_adamw.yaml`:

```yaml
name: muon_vs_adamw
display_keys: [optimizer, lr, seed]

base:
  dataset: mnist
  model: mlp
  rule: bp
  epochs: 20
  batch: 64

grid:
  optimizer: [muon, adamw]
  lr: [0.02, 0.01, 0.001]
  seed: [0, 1, 2, 3, 4]
```

That gives you `2 optimizers × 3 learning rates × 5 seeds = 30` runs. Each optimizer gets a fair shot at its best learning rate, and you have enough seeds to see variance.

Preview first:

```bash
lelabo sweep run --config sweeps/muon_vs_adamw.yaml --dry-run
```

Then run:

```bash
lelabo sweep run --config sweeps/muon_vs_adamw.yaml --max-parallel 4
```

## Step 7 — Track results in W&B

If you have W&B set up (see the [W&B guide](../guides/wandb.md) for installation and login), enable it before launching the sweep:

```bash
export WANDB_PROJECT=muon-paper

lelabo sweep run --config sweeps/muon_vs_adamw.yaml --max-parallel 4
```

All 30 runs are automatically grouped under `muon_vs_adamw` in the W&B dashboard. From there:

- filter by `config.optimizer` to overlay Muon runs against AdamW runs
- use the parallel coordinates view to see which learning rate works best for each optimizer
- sort by `summary.best.best_value` to find the best configuration overall

If you don't use W&B, the same information is available locally. Every run writes `summary.json` and `metrics.jsonl` in its output directory. A quick shell loop can extract what you need:

```bash
for dir in outputs/runs/muon_vs_adamw/*/; do
  name=$(basename "$dir")
  acc=$(python -c "
import json
s = json.load(open('${dir}summary.json'))
print(f\"{s['best']['best_value']:.4f}\")
" 2>/dev/null || echo "N/A")
  echo "$name → $acc"
done
```

## Step 8 — Document and pack

Update the capsule's `README.md` to document:

- What the paper claims
- What you implemented (the optimizer only, not the full paper setup)
- What hyperparameters you used vs the paper's values
- Your results vs the baseline

Then pack:

```bash
lelabo capsule pack \
  --from outputs/muon_mnist \
  --out muon_paper_v1.tar.gz
```

Anyone can now install and reproduce:

```bash
lelabo capsule install muon_paper_v1.tar.gz
```

## What makes a good paper pack

A paper pack is most useful when it is honest:

- **What's implemented** — "We implement the Muon optimizer. Model and training loop are LeLabo builtins."
- **What's not** — "We did not implement the learning rate warmup schedule described in Section 4.2."
- **Hyperparameters** — which values were taken from the paper, which were tuned
- **Results** — your numbers vs the paper's, on which datasets
- **Known differences** — any divergence from the original setup

## Something missing in the builtins?

If you find that a builtin model, dataset, or loss doesn't quite support what the paper needs — don't silently reimplement it. [Open an issue](https://github.com/adrienkegreisz/LeLabo/issues) so we can improve the shared component for everyone.

The goal is that the community shares the same reference environment. The more people use the same builtins, the more comparable experiments become.
