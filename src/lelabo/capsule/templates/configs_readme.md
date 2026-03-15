# <capsule_name>

   This capsule is a local LeLabo scaffold for implementing a paper, method, or
   custom extension.
   
   Use this README as a short entry point. Once the capsule becomes a real project,
   rewrite this file so it documents your own method, configs, assumptions, and
   results.
   
   ## Quick start
   
   First path to a working implementation:
   
   1. Uncomment `@register_optimizer("capsule_sgd")` in `optimizers/example.py`
   2. Run `lelabo list optimizers`
   3. Run `lelabo train supervised --config configs/train.supervised.capsule_optimizer.toml`
   4. Run `pytest -q tests`
   
   ## How to navigate this capsule
   
   - `AGENTS.md`
     Main local guide for Codex/Claude and task routing
   
   - `resources/EXTENSION_RECIPES.md`
     Short procedural guide for adding a model, update rule, dataset, optimizer, or paper pack
   
   - `resources/LELABO_REFERENCE.md`
     Exact builder signatures, contexts, and public contracts
   
   - `resources/PARAM_FLOW.md`
     Where config params arrive at runtime
   
   - `resources/MODEL_CACHE_ADVANCED.md`
     Only if your extension needs intermediate activations, block semantics, or local-rule cache support
   
   - `resources/UPDATE_RULE_LIFECYCLE.md`
     Runtime contract for custom update rules
   
   - `resources/PAPER_PACK_PLAYBOOK.md`
     How to structure a multi-component paper implementation cleanly
   
   ## Human docs
   
   For the longer human-oriented LeLabo documentation, see:
   
   `https://adrienkegreisz.github.io/lelabo-docs`
   
   ## When to rewrite this README
   
   Rewrite this file once the capsule becomes your real project README.
   
   At that point, this file should describe:
   
   - what your paper or method does
   - which components in the capsule are custom
   - which configs are the main entry points
   - how to validate the implementation
   - important assumptions or runtime limits
   - expected results or current status
   