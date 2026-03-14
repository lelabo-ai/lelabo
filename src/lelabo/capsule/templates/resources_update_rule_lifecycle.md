# Update Rule Lifecycle

This file explains the runtime contract of an update rule.

It is a framework contract document, not a recipe for a specific algorithm.

## What an update rule is

An update rule owns the training logic for a step.

Typical responsibilities:

- decide how to use the model
- decide whether to use cache collection
- compute or assign parameter updates
- return numeric training stats

## Core runtime shape

The rule entrypoint is conceptually:

```python
train_step(model, objective, batch, device, state=None) -> dict[str, Any]
```

## What the rule receives

- `model`
  the current `nn.Module`

- `objective`
  the loss/runtime objective object already assembled by LeLabo

- `batch`
  training batch

- `device`
  target device for the step

- `state`
  current training state when available

## What the rule may do

An update rule may:

- run a standard BP step
- call `forward_with_standard_cache(...)`
- inspect `state`
- freeze or unfreeze model parameters
- update only part of the model
- use a custom optimizer behavior
- return custom scalar stats

## What the rule must return

Return a dictionary of numeric scalars.

Typical keys:

- `loss`
- `acc`
- `loss_proxy`
- other scalar diagnostics

Avoid returning:

- tensors
- nested arbitrary objects
- non-numeric values

## Using `state`

`state` is where the rule can read runtime progression.

Typical examples:

- current epoch
- current step
- global step

Use it to condition rule behavior when needed.

Do not assume a richer staged-program abstraction exists today.

## Rule vs model vs dataset

Keep these responsibilities separate:

- model
  structure and forward behavior

- update rule
  how parameters are updated

- dataset
  how data and splits are loaded

Do not hide dataset orchestration inside a model if it is really experiment logic.

## Relation to optimizer / loss / cache

- use the optimizer given by the runner unless your design explicitly needs something else
- use `forward_with_standard_cache(...)` only when the rule needs internal activations
- prefer the smallest `CacheSpec` that covers the rule’s needs

## Common failure modes

- the rule returns non-numeric stats
- the rule assumes a block structure the model does not expose
- the rule expects several views at once instead of requesting a single `target_view`
- the rule assumes staged multi-phase runtime support that does not exist yet

## Current runtime limits

The supervised runtime is strong for:

- standard BP
- many local-rule setups
- rules that can express their logic in a normal step loop

It is not yet first-class for:

- general staged multi-phase experiment programs
- several loaders coordinated as explicit runtime phases
- arbitrary pretrain/freeze/fine-tune programs as a public runtime object

If your rule needs that level of orchestration, document the workaround clearly in the capsule.
