# Research notes

## Project status

LeLabo is under active development.

Current priorities are:

- making research loops faster to run
- keeping models close to plain PyTorch
- supporting local learning rules through a common cache contract
- reducing framework-specific boilerplate for custom experiments

## Current design choices

### Plain `nn.Module` models

Custom models are expected to stay close to normal PyTorch modules.

That means:

- regular `forward(...)`
- standard submodules
- minimal framework-specific inheritance

### Registry-based composition

The CLI composes experiments from registries:

- dataset
- model
- initializer
- loss
- update rule
- optimizer
- scheduler

This makes ablations easy without rewriting training code.

### Cache-first support for local rules

Local update rules rely on the standard cache provider.

In practice, this means:

- using `nn.Module` activations is strongly preferred
- `torch.nn.functional.*` activations are still usable for backprop, but reduce cache quality for local rules
- `get_blocks()` is useful for models with natural high-level blocks, such as ResNets or Hugging Face models

## Current limitations

- APIs are not yet stable
- docs are intentionally minimal for now
- some rules are still evolving quickly
- performance and ergonomics are secondary to conceptual correctness

## Recommended mindset

Use LeLabo as a research workbench:

- start from a built-in baseline
- inspect the cache if you work on local rules
- add custom components through capsules
- expect some internal refactors over time


## Cache : 

selection

quels modules/blocs on observe
ex: trainable_module_types, observed_module_types, observed_module_names, block_source
views

quelles représentations dérivées on veut obtenir
ex: include_local_blocks, include_model_blocks, auto_pair_post_activation
contract

quelles hypothèses la règle impose sur le cache
ex: require_single_call, require_single_output_head, require_input_ndim, require_output_ndim