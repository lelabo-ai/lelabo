# Build a Paper Pack

A paper pack is a capsule that holds a complete method implementation — model, update rule, config, and tests — in a form that can be shared, reinstalled, and compared against other methods without rebuilding anything.

## What belongs in a paper pack

| Component | Description |
|---|---|
| Custom model | The architecture used in the paper |
| Custom update rule | The learning rule or credit assignment scheme |
| Custom dataset | If the method requires specific data loading |
| Configs | Ready-to-run configs reproducing the paper results |
| Tests | Smoke tests that verify the method runs correctly |
| Documentation | Notes on what the method does and what the paper claims |

A paper pack does not need to contain all of these. Start with what changes — usually a model + update rule + config.

## Start from the scaffold

```bash
lelabo capsule init my_paper
cd my_paper
```

The scaffold includes a `configs/train/supervised.paper_pack.toml` that is already set up as a starting point for paper pack experiments.

## Structure your method

A well-structured paper pack typically follows this layout:

```
my_paper/
├── models/
│   └── my_arch.py          ← the paper's architecture
├── update_rules/
│   └── my_rule.py          ← the paper's learning rule
├── configs/
│   └── train/supervised.paper_pack.toml
├── tests/
│   └── test_paper_pack_smoke.py
└── resources/
    └── PAPER_PACK_PLAYBOOK.md   ← detailed local guide
```

!!! tip "Read the local playbook first"
    `resources/PAPER_PACK_PLAYBOOK.md` inside the capsule has a detailed workflow for paper pack implementation. It covers edge cases, testing strategy, and how to document runtime assumptions.

## Register your components

Follow the same pattern as the individual guides:

```python
# models/my_arch.py
from lelabo.models.registry import ModelContext, register_model

@register_model("paper_model")
def build_paper_model(ctx: ModelContext, args): ...
```

```python
# update_rules/my_rule.py
from lelabo.update_rules.registry import UpdateRuleContext, register_update_rule

@register_update_rule("paper_rule")
def build_paper_rule(ctx: UpdateRuleContext): ...
```

## Wire the config

Point the paper pack config at your components:

```toml
[model]
name = "paper_model"
[model.params]
hidden = 256

[update_rule]
name = "paper_rule"
[update_rule.params]
grad_clip = 1.0

[dataset]
name = "mnist"
```

## Run and verify

```bash
lelabo list models       # paper_model should appear
lelabo list update-rules # paper_rule should appear

lelabo train supervised \
  --config configs/train/supervised.paper_pack.toml \
  --run-dir outputs/paper_run
```

```bash
pytest -q tests/test_paper_pack_smoke.py
```

## Pack and share

Once the method runs correctly, build a shareable bundle:

```bash
lelabo capsule pack \
  --from outputs/paper_run \
  --out my_paper_v1.tar.gz
```

The bundle contains the capsule code, the run artifacts, and the resolved config. Anyone can install it:

```bash
lelabo capsule install my_paper_v1.tar.gz
lelabo capsule checkout my_paper
```

## What a good paper pack documents

A paper pack is most useful when it is honest about what it implements:

- Which parts of the paper are implemented
- Which hyperparameters were tuned vs taken from the paper
- What datasets and settings were tested
- Known differences from the original results
- Any runtime assumptions (e.g. requires `declare_blocks()`)

This documentation lives in the capsule's `README.md` and in the config comments.
