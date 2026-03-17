# Implement DFA from Scratch

This tutorial walks through implementing Direct Feedback Alignment (DFA) as a custom update rule, step by step. It covers the cache contract, `declare_blocks()`, and the full local-rule workflow.

By the end, you'll have a working DFA rule that you can run on MNIST and compare against standard backpropagation.

## What is DFA?

In standard backpropagation, error gradients flow backwards through the network layer by layer, each layer using the transpose of its own weight matrix. This creates a dependency chain — every layer must wait for the layer above.

**Direct Feedback Alignment** (Nøkland, 2016) replaces this chain with direct random projections. Each hidden layer receives the output error directly through a fixed random matrix, bypassing all intermediate layers:

```
Standard BP:    output → layer_3 → layer_2 → layer_1
DFA:            output → layer_1  (via random matrix B₁)
                output → layer_2  (via random matrix B₂)
                output → layer_3  (via random matrix B₃)
```

The random matrices are never trained — they stay fixed. Surprisingly, this works. Not as well as BP on every task, but well enough to be interesting for research on credit assignment.

## Step 1 — Create the capsule

```bash
lelabo capsule init dfa_tutorial
cd dfa_tutorial
```

## Step 2 — Build a model with `declare_blocks()`

DFA needs to know the layer structure of the model — it has to attach a random feedback matrix to each hidden layer. The cache contract provides this through `declare_blocks()`.

Open `models/example.py`:

```python
import torch.nn as nn
from lelabo.models.registry import ModelContext, register_model
from lelabo.models.blocks import BlockSpec


@register_model("dfa_mlp")
def build_dfa_mlp(ctx: ModelContext, args):
    params = dict(getattr(args, "model_params", {}) or {})
    hidden = params.get("hidden", 256)

    class DFAMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(ctx.in_dim, hidden)
            self.act1 = nn.ReLU()
            self.layer2 = nn.Linear(hidden, hidden)
            self.act2 = nn.ReLU()
            self.output = nn.Linear(hidden, ctx.num_classes)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.act1(self.layer1(x))
            x = self.act2(self.layer2(x))
            return self.output(x)

        def declare_blocks(self) -> list[BlockSpec]:           # (1)
            return [
                BlockSpec(name="layer1", module=self.layer1),  # (2)
                BlockSpec(name="layer2", module=self.layer2),
                BlockSpec(
                    name="output",
                    module=self.output,
                    is_output=True,                            # (3)
                ),
            ]

    return DFAMLP()
```

Key points:

1. `declare_blocks()` returns a list of `BlockSpec` — this tells the cache which layers exist and their order.
2. Each `BlockSpec` gives a name and a reference to the `nn.Module`. The cache will hook onto these modules to capture inputs and outputs during the forward pass.
3. `is_output=True` marks the final layer. DFA uses this to compute the output error delta.

!!! tip "Use builtins if they support your model"
    The builtin `mlp` already implements `declare_blocks()`. If you're just experimenting with DFA on a standard MLP, use `--model mlp` and skip this step. We implement it here for pedagogical purposes.

## Step 3 — Implement the DFA update rule

Open `update_rules/example.py`:

```python
import torch
import torch.nn as nn
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule
from lelabo.models.cache_provider import CacheSpec, forward_with_standard_cache


@register_update_rule("dfa_tutorial")
def build_dfa_tutorial(ctx: UpdateRuleContext):
    params = ctx.extra.get("update_rule_params", {}) or {}
    feedback_scale = params.get("feedback_scale", 1.0)

    # Define what we need from the cache
    cache_spec = CacheSpec(                                      # (1)
        target_view="execution",
        trainable_module_types=(nn.Linear,),
        capture_inputs=True,
        capture_outputs=True,
        require_single_call=True,
    )

    class DFATutorial:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer
            self.feedback = {}                                   # (2)

        def _ensure_feedback(self, name, in_features, out_features, device):
            """Create a fixed random feedback matrix if it doesn't exist yet."""
            if name not in self.feedback:
                B = torch.randn(out_features, in_features, device=device)
                B *= feedback_scale / (in_features ** 0.5)
                self.feedback[name] = B

        def train_step(self, batch, state):
            x, y = batch
            self.optimizer.zero_grad()

            # Forward pass with cache collection
            out, _cache, views = forward_with_standard_cache(   # (3)
                self.model, x, cache_spec=cache_spec
            )
            execution_blocks = views.get("execution", [])

            # Compute loss and output error
            loss = ctx.loss_fn(out, y)
            num_classes = out.shape[-1]

            # One-hot targets for computing the output delta
            if y.dim() == 1:
                y_onehot = torch.zeros(y.size(0), num_classes, device=y.device)
                y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            else:
                y_onehot = y

            delta_output = out.softmax(dim=-1) - y_onehot       # (4)

            # Find the output block
            output_block = None
            hidden_blocks = []
            for block in execution_blocks:
                if block.get("is_output", False):
                    output_block = block
                else:
                    if block.get("is_trainable", False):
                        hidden_blocks.append(block)

            # Update output layer with standard gradient
            if output_block is not None:
                x_out = output_block.get("x")                   # (5)
                if x_out is not None and x_out.dim() == 2:
                    module = output_block.get("module")
                    module.weight.grad = (delta_output.T @ x_out) / x.size(0)
                    if module.bias is not None:
                        module.bias.grad = delta_output.mean(dim=0)

            # Update hidden layers with direct feedback
            for block in hidden_blocks:                          # (6)
                name = block.get("name", "")
                module = block.get("module")
                x_in = block.get("x")       # input to this layer
                u = block.get("u")          # output of this layer (pre-activation)

                if x_in is None or u is None:
                    continue
                if x_in.dim() != 2 or u.dim() != 2:
                    continue

                out_features = u.shape[-1]
                num_outputs = delta_output.shape[-1]
                self._ensure_feedback(name, num_outputs, out_features, u.device)

                # DFA: project output error directly to this layer
                B = self.feedback[name]
                delta_local = delta_output @ B.T                 # (7)

                # Multiply by activation derivative (ReLU)
                act_mask = (u > 0).float()
                delta_local = delta_local * act_mask

                # Assign gradients
                module.weight.grad = (delta_local.T @ x_in) / x.size(0)
                if module.bias is not None:
                    module.bias.grad = delta_local.mean(dim=0)

            self.optimizer.step()

            return {"loss": loss.item()}

    return DFATutorial
```

Let's break down the key parts:

1. **`CacheSpec`** — declares what we need. `target_view="execution"` gives us blocks in forward-pass order. `trainable_module_types=(nn.Linear,)` tells the cache to focus on linear layers. `capture_inputs=True` and `capture_outputs=True` give us the tensors we need for computing updates.

2. **`self.feedback`** — stores the fixed random matrices. Created once per layer, never updated.

3. **`forward_with_standard_cache()`** — runs the forward pass and collects everything specified in the `CacheSpec`. Returns the model output, a raw cache dict, and the requested views.

4. **`delta_output`** — the error at the output layer. For cross-entropy + softmax, this is simply `softmax(logits) - one_hot_targets`.

5. **`block.get("x")`** — the input tensor captured by the cache. `block.get("u")` is the output (pre-activation). These are the tensors we need to compute local weight updates.

6. **Hidden layer loop** — this is where DFA differs from BP. Instead of backpropagating through each layer, we project the output error directly to each hidden layer through a fixed random matrix.

7. **`delta_output @ B.T`** — the direct feedback projection. `B` is a random matrix of shape `(out_features, num_outputs)`. This projects the output-layer error directly to the hidden layer, bypassing all intermediate layers.

## Step 4 — Configure and run

Edit `configs/train.supervised.paper_pack.toml`:

```toml
config_version = "auto"
lelabo_version = "auto"
task = "supervised"

[dataset]
name = "mnist"

[model]
name = "dfa_mlp"
[model.params]
hidden = 256

[loss]
name = "ce"

[update_rule]
name = "dfa_tutorial"
[update_rule.params]
feedback_scale = 1.0

[optimizer]
name = "sgd"
[optimizer.params]
lr = 0.01
momentum = 0.9

[runtime]
device = "auto"
seed = 42
run_dir = "outputs/dfa_mnist"

[train]
epochs = 20
batch = 64

[[callbacks]]
name = "earlystopping"
enabled = true
[callbacks.params]
monitor = "val.acc"
patience = 5

[[metrics]]
name = "acc"
```

```bash
lelabo train supervised --config configs/train.supervised.paper_pack.toml
```

## Step 5 — Compare with BP baseline

```bash
lelabo train supervised \
  --config configs/train.supervised.paper_pack.toml \
  --model mlp \
  --rule bp \
  --optimizer adamw \
  --lr 0.001 \
  --run-dir outputs/bp_mnist
```

Compare:

```bash
python -c "
import json
for name, path in [('DFA', 'outputs/dfa_mnist'), ('BP', 'outputs/bp_mnist')]:
    with open(f'{path}/summary.json') as f:
        s = json.load(f)
    best = s['best']
    print(f\"{name}: best {best['monitor_name']} = {best['best_value']:.4f} at epoch {best['epoch']}\")
"
```

DFA will typically reach 95-97% on MNIST. BP will reach 97-98%. The gap is expected — DFA trades some accuracy for a biologically more plausible credit assignment scheme.

## Step 6 — Test with `lelabo audit`

```bash
lelabo audit --rule dfa_tutorial
```

The audit verifies that the rule runs without errors across standard setups and produces reasonable outputs.

## Debugging a local rule

Common issues:

| Symptom | Likely cause |
|---|---|
| `KeyError: "execution"` | Model doesn't implement `declare_blocks()`, or `CacheSpec` is misconfigured |
| All blocks have `x=None` | `capture_inputs=False` in `CacheSpec` |
| Loss stays flat | Feedback matrices are too large/small — adjust `feedback_scale` |
| `RuntimeError: shape mismatch` | Block dimensions don't match — check `in_features` vs `out_features` |
| Only output layer learns | Hidden blocks are not being iterated — check the `is_output` / `is_trainable` filtering |

To inspect what the cache captures:

```python
# Inside train_step, after forward_with_standard_cache:
for block in execution_blocks:
    print(f"{block.get('name'):10s}  trainable={block.get('is_trainable')}  "
          f"x={block.get('x') is not None}  u={block.get('u') is not None}")
```

## Next steps

- [Compare update rules](compare-rules.md) — run BP, DFA, and FA on the same setup and compare
- [Cache & local rules](../concepts/cache-local-rules.md) — the full contract reference
- [Create an update rule](../guides/create-update-rule.md) — the concise guide version
