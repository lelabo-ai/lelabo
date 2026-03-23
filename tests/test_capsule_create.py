from __future__ import annotations

import importlib
import json
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
create = importlib.import_module("lelabo.capsule.create")
registry = importlib.import_module("lelabo.capsule.registry")
plugins = importlib.import_module("lelabo.capsule.plugins")
models_registry = importlib.import_module("lelabo.models.registry")
datasets_registry = importlib.import_module("lelabo.supervised.datasets.registry")
train_api = importlib.import_module("lelabo.cli.commands.train")
capsule_cli = importlib.import_module("lelabo.cli.commands.capsule")
list_cli = importlib.import_module("lelabo.cli.commands.list")
lab_pkg = importlib.import_module("lelabo")


def test_create_capsule_scaffold_creates_expected_layout(tmp_path) -> None:
    capsules_dir = tmp_path / ".lelabo" / "capsules"
    out = create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=tmp_path,
        capsules_dir=capsules_dir,
        register=True,
    )

    assert out == (tmp_path / "demo_capsule")
    assert out.is_dir()
    assert (out / "models").is_dir()
    assert (out / "update_rules").is_dir()
    assert (out / "datasets").is_dir()
    assert (out / "metrics").is_dir()
    assert (out / "initializers").is_dir()
    assert (out / "losses").is_dir()
    assert (out / "optimizers").is_dir()
    assert (out / "schedulers").is_dir()
    assert (out / "callbacks").is_dir()
    assert (out / "configs").is_dir()
    assert (out / "runs").is_dir()
    assert (out / "resources").is_dir()
    assert (out / "tests").is_dir()
    assert (out / "README.md").exists()
    assert (out / "AGENTS.md").exists()
    assert (out / "capsule.toml").exists()
    assert (out / "manifest.json").exists()
    assert (out / "models" / "__init__.py").exists()
    assert (out / "models" / "example.py").exists()
    assert (out / "models" / "cache_walkthrough.py").exists()
    assert (out / "resources" / "LELABO_REFERENCE.md").exists()
    assert (out / "resources" / "PARAM_FLOW.md").exists()
    assert (out / "resources" / "MODEL_CACHE_ADVANCED.md").exists()
    assert (out / "resources" / "UPDATE_RULE_LIFECYCLE.md").exists()
    assert (out / "resources" / "PAPER_PACK_PLAYBOOK.md").exists()
    assert (out / "resources" / "EXTENSION_RECIPES.md").exists()
    assert (out / "update_rules" / "example.py").exists()
    assert (out / "datasets" / "example.py").exists()
    assert (out / "metrics" / "example.py").exists()
    assert (out / "initializers" / "initializer_helpers.py").exists()
    assert (out / "initializers" / "example.py").exists()
    assert (out / "losses" / "example.py").exists()
    assert (out / "optimizers" / "example.py").exists()
    assert (out / "schedulers" / "example.py").exists()
    assert (out / "callbacks" / "example.py").exists()
    assert (out / "configs" / "README.md").exists()
    assert (out / "configs" / "train" / "supervised.quickstart.toml").exists()
    assert (out / "configs" / "train" / "supervised.detailed.toml").exists()
    assert (out / "configs" / "train" / "supervised.capsule_optimizer.toml").exists()
    assert (out / "configs" / "train" / "supervised.paper_pack.toml").exists()
    assert (out / "configs" / "train" / "rl.detailed.toml").exists()
    assert (out / "runs" / "example.py").exists()
    assert (out / "tests" / "test_capsule_optimizer_smoke.py").exists()
    assert (out / "tests" / "test_paper_pack_smoke.py").exists()
    model_example = (out / "models" / "example.py").read_text(encoding="utf-8")
    model_walkthrough = (out / "models" / "cache_walkthrough.py").read_text(encoding="utf-8")
    metric_example = (out / "metrics" / "example.py").read_text(encoding="utf-8")
    initializer_helper = (out / "initializers" / "initializer_helpers.py").read_text(encoding="utf-8")
    initializer_example = (out / "initializers" / "example.py").read_text(encoding="utf-8")
    loss_example = (out / "losses" / "example.py").read_text(encoding="utf-8")
    optimizer_example = (out / "optimizers" / "example.py").read_text(encoding="utf-8")
    scheduler_example = (out / "schedulers" / "example.py").read_text(encoding="utf-8")
    callback_example = (out / "callbacks" / "example.py").read_text(encoding="utf-8")
    readme_text = (out / "README.md").read_text(encoding="utf-8")
    agents_text = (out / "AGENTS.md").read_text(encoding="utf-8")
    extension_recipes = (out / "resources" / "EXTENSION_RECIPES.md").read_text(encoding="utf-8")
    lelabo_reference = (out / "resources" / "LELABO_REFERENCE.md").read_text(encoding="utf-8")
    param_flow = (out / "resources" / "PARAM_FLOW.md").read_text(encoding="utf-8")
    model_cache_advanced = (out / "resources" / "MODEL_CACHE_ADVANCED.md").read_text(encoding="utf-8")
    update_rule_lifecycle = (out / "resources" / "UPDATE_RULE_LIFECYCLE.md").read_text(encoding="utf-8")
    paper_pack_playbook = (out / "resources" / "PAPER_PACK_PLAYBOOK.md").read_text(encoding="utf-8")
    smoke_test = (out / "tests" / "test_capsule_optimizer_smoke.py").read_text(encoding="utf-8")
    paper_pack_smoke = (out / "tests" / "test_paper_pack_smoke.py").read_text(encoding="utf-8")
    capsule_toml = (out / "capsule.toml").read_text(encoding="utf-8")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    configs_readme = (out / "configs" / "README.md").read_text(encoding="utf-8")
    cfg_quick = (out / "configs" / "train" / "supervised.quickstart.toml").read_text(encoding="utf-8")
    cfg_paper = (out / "configs" / "train" / "supervised.paper_pack.toml").read_text(encoding="utf-8")
    assert "register_model" in model_example
    assert 'Uncomment `@register_model("example_mlp")`' in model_example
    assert "cache_walkthrough.py" in model_example
    assert "forward_with_standard_cache" in model_walkthrough
    assert "register_metric" in metric_example
    assert "ClassificationStreamingMetric" in metric_example
    assert "register_metric_fn" not in metric_example
    assert '# @register_metric("example_error_rate", kind="classification")' in metric_example
    assert "make_initializer" in initializer_helper
    assert "from lelabo.initializers.registry import InitializerContext" in initializer_helper
    assert "register_initializer" in initializer_example
    assert 'register_initializer("row_sum_one")' in initializer_example
    assert "from .initializer_helpers import make_initializer" in initializer_example
    assert "register_loss" in loss_example
    assert 'register_loss("example_scaled_l1")' in loss_example
    assert "# @register_loss" in loss_example
    assert "from lelabo.optimizers import OptimizerContext, register_optimizer" in optimizer_example
    assert 'Uncomment `@register_optimizer("capsule_sgd")`' in optimizer_example
    assert "from lelabo.schedulers import SchedulerContext, register_scheduler" in scheduler_example
    assert "# @register_scheduler" in scheduler_example
    assert "from lelabo.callbacks import Callback" in callback_example
    assert "CallbackContext" in callback_example
    assert "register_callback" in callback_example
    assert "# @register_callback" in callback_example
    assert "AGENTS.md" in readme_text
    assert "capsule_sgd" in readme_text
    assert "pytest -q tests" in readme_text
    assert "resources/LELABO_REFERENCE.md" not in readme_text
    assert "resources/PARAM_FLOW.md" not in readme_text
    assert "If your idea is X, edit Y" in agents_text
    assert "Decision Guide" in agents_text
    assert "Which file answers which question?" in agents_text
    assert "Stop Signs" in agents_text
    assert "matching `example.py`" in agents_text
    assert "config changes before custom code" in agents_text
    assert "training logic or credit assignment" in agents_text
    assert "resources/EXTENSION_RECIPES.md" in agents_text
    assert "resources/LELABO_REFERENCE.md" in agents_text
    assert "resources/PARAM_FLOW.md" in agents_text
    assert "resources/MODEL_CACHE_ADVANCED.md" in agents_text
    assert "resources/UPDATE_RULE_LIFECYCLE.md" in agents_text
    assert "resources/PAPER_PACK_PLAYBOOK.md" in agents_text
    assert "`optimizers/`" in agents_text
    assert "pytest -q tests" in agents_text
    assert "Choose the right extension type first" in extension_recipes
    assert "normal capsule work" in extension_recipes
    assert "Add an optimizer" in extension_recipes
    assert "Add a BP model" in extension_recipes
    assert "Add a local-rule update rule" in extension_recipes
    assert "Add a dataset" in extension_recipes
    assert "Add a paper pack" in extension_recipes
    assert "Short appendix: secondary extension types" in extension_recipes
    assert "Core done criteria" in extension_recipes
    assert "Extra done criteria" in extension_recipes
    assert "appears in `lelabo list" in extension_recipes
    assert "ModelContext" in lelabo_reference
    assert "OptimizerContext" in lelabo_reference
    assert "SchedulerContext" in lelabo_reference
    assert "CallbackContext" in lelabo_reference
    assert "MetricContext" in lelabo_reference
    assert "LossContext" in lelabo_reference
    assert "InitializerContext" in lelabo_reference
    assert "UpdateRuleContext" in lelabo_reference
    assert "DataBundle" in lelabo_reference
    assert "forward_with_standard_cache" in lelabo_reference
    assert "resolve_declared_blocks" in lelabo_reference
    assert "register_cache_pair_activation" in lelabo_reference
    assert "not the first file to read" in lelabo_reference
    assert "`forward_with_standard_cache(...)` returns `out`, `cache`, and `views`" in lelabo_reference
    assert "contains only the requested `CacheSpec.target_view`, not all public views at once" in lelabo_reference
    assert "[model.params] -> args.model_params" in param_flow
    assert "[optimizer.params] -> ctx.optimizer_params()" in param_flow
    assert "[scheduler.params] -> ctx.scheduler_params()" in param_flow
    assert "[loss.params] -> ctx.loss_params()" in param_flow
    assert "[initializer.params] -> ctx.initializer_params()" in param_flow
    assert "[callbacks.params] -> ctx.callback_params()" in param_flow
    assert "[metrics.params.<metric_name>] -> ctx.metric_params(name)" in param_flow
    assert '[update_rule.params] -> ctx.extra["update_rule_params"]' in param_flow
    assert "Debugging param issues" in param_flow
    assert "CacheSpec" in model_cache_advanced
    assert "ResolvedBlock" in model_cache_advanced
    assert "target_view" in model_cache_advanced
    assert "exec_module" in model_cache_advanced
    assert "exec_span_names" in model_cache_advanced
    assert "register_cache_pair_activation" in model_cache_advanced
    assert "`forward_with_standard_cache(...)` returns `out`, `cache`, and `views`" in model_cache_advanced
    assert "contains only the requested `CacheSpec.target_view`, not all public views at once" in model_cache_advanced
    assert "forward_with_standard_cache" in model_cache_advanced
    assert "Do not open this" in model_cache_advanced
    assert "`declare_blocks()` is optional" in model_cache_advanced
    assert "train_step(model, objective, batch, device, state=None)" in update_rule_lifecycle
    assert "Tiny pseudo-template" in update_rule_lifecycle
    assert "What the rule must return" in update_rule_lifecycle
    assert "Current runtime limits" in update_rule_lifecycle
    assert "paper pack" in paper_pack_playbook.lower()
    assert "Implementation order" in paper_pack_playbook
    assert "Validation checklist" in paper_pack_playbook
    assert "When not to build a paper pack" in paper_pack_playbook
    assert 'Uncomment `@register_optimizer("capsule_sgd")` in `optimizers/example.py`' in smoke_test
    assert 'CONFIG_PATH = CAPSULE_ROOT / "configs" / "train" / "supervised.capsule_optimizer.toml"' in smoke_test
    assert 'Uncomment `@register_model("example_mlp")`' in paper_pack_smoke
    assert 'Uncomment `@register_update_rule("local_head")`' in paper_pack_smoke
    assert 'CONFIG_PATH = CAPSULE_ROOT / "configs" / "train" / "supervised.paper_pack.toml"' in paper_pack_smoke
    assert "lelabo_version" in capsule_toml
    assert str(manifest.get("lelabo_version", "")).strip()
    assert 'config_version = "1.0"' in cfg_quick
    assert 'config_version = "1.0"' in cfg_paper
    assert 'name = "example_mlp"' in cfg_paper
    assert 'name = "local_head"' in cfg_paper
    assert f'lelabo_version = "{lab_pkg.__version__}"' in cfg_quick
    assert f'lelabo_version = "{lab_pkg.__version__}"' in cfg_paper
    assert 'config_version = "auto"' not in cfg_quick
    assert "AGENTS.md" in configs_readme
    assert "train/supervised.quickstart.toml" in configs_readme
    assert "train/supervised.capsule_optimizer.toml" in configs_readme
    assert "train/supervised.paper_pack.toml" in configs_readme
    assert "optimizers/example.py" in configs_readme
    assert "models/cache_walkthrough.py" in configs_readme
    assert "../resources/EXTENSION_RECIPES.md" in configs_readme
    assert "../resources/LELABO_REFERENCE.md" in configs_readme
    assert "../resources/PARAM_FLOW.md" in configs_readme
    assert "../resources/MODEL_CACHE_ADVANCED.md" in configs_readme
    assert "../resources/UPDATE_RULE_LIFECYCLE.md" in configs_readme
    assert "../resources/PAPER_PACK_PLAYBOOK.md" in configs_readme
    assert "resources/EXTENSION_RECIPES.md" in model_example
    assert "resources/LELABO_REFERENCE.md" in model_example
    assert "resources/PARAM_FLOW.md" in model_example
    assert "resources/PAPER_PACK_PLAYBOOK.md" in model_example
    assert "resources/EXTENSION_RECIPES.md" in optimizer_example
    assert "resources/LELABO_REFERENCE.md" in optimizer_example
    assert "resources/PARAM_FLOW.md" in optimizer_example
    assert "resources/EXTENSION_RECIPES.md" in metric_example
    assert "resources/LELABO_REFERENCE.md" in metric_example
    assert "resources/PARAM_FLOW.md" in metric_example
    assert "resources/EXTENSION_RECIPES.md" in initializer_example
    assert "resources/LELABO_REFERENCE.md" in initializer_example
    assert "resources/PARAM_FLOW.md" in initializer_example
    assert "resources/EXTENSION_RECIPES.md" in loss_example
    assert "resources/LELABO_REFERENCE.md" in loss_example
    assert "resources/PARAM_FLOW.md" in loss_example
    assert "resources/EXTENSION_RECIPES.md" in scheduler_example
    assert "resources/LELABO_REFERENCE.md" in scheduler_example
    assert "resources/PARAM_FLOW.md" in scheduler_example
    assert "resources/EXTENSION_RECIPES.md" in callback_example
    assert "resources/LELABO_REFERENCE.md" in callback_example
    assert "resources/PARAM_FLOW.md" in callback_example
    assert "resources/EXTENSION_RECIPES.md" in (out / "datasets" / "example.py").read_text(encoding="utf-8")
    assert "resources/LELABO_REFERENCE.md" in (out / "datasets" / "example.py").read_text(encoding="utf-8")
    assert "resources/PARAM_FLOW.md" in (out / "datasets" / "example.py").read_text(encoding="utf-8")
    assert "resources/EXTENSION_RECIPES.md" in (out / "update_rules" / "example.py").read_text(encoding="utf-8")
    assert "resources/LELABO_REFERENCE.md" in (out / "update_rules" / "example.py").read_text(encoding="utf-8")
    assert "resources/PARAM_FLOW.md" in (out / "update_rules" / "example.py").read_text(encoding="utf-8")
    assert "resources/MODEL_CACHE_ADVANCED.md" in (out / "update_rules" / "example.py").read_text(encoding="utf-8")
    assert "resources/UPDATE_RULE_LIFECYCLE.md" in (out / "update_rules" / "example.py").read_text(encoding="utf-8")
    assert "resources/PAPER_PACK_PLAYBOOK.md" in (out / "update_rules" / "example.py").read_text(encoding="utf-8")
    row = registry.get_capsule("demo_capsule", capsules_dir)
    assert row is not None
    assert row["capsule_id"] == "demo_capsule"
    assert row["source_bundle"] == "local_scaffold"


def test_create_capsule_scaffold_requires_force_for_existing_dir(tmp_path) -> None:
    target = tmp_path / "existing_capsule"
    target.mkdir()

    with pytest.raises(FileExistsError):
        create.create_capsule_scaffold(capsule_name="existing_capsule", base_dir=tmp_path, register=False)

    out = create.create_capsule_scaffold(capsule_name="existing_capsule", base_dir=tmp_path, force=True, register=False)
    assert out == target
    assert (target / "models").is_dir()


def test_create_capsule_scaffold_rejects_invalid_name(tmp_path) -> None:
    with pytest.raises(ValueError):
        create.create_capsule_scaffold(capsule_name="bad/name", base_dir=tmp_path)


def test_scaffold_examples_are_train_resolvable(tmp_path, monkeypatch) -> None:
    out = create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=tmp_path,
        register=False,
    )

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    original_dataset_items = dict(datasets_registry.DATASET_REGISTRY._items)
    models_registry.MODEL_REGISTRY._items = {}
    datasets_registry.DATASET_REGISTRY._items = {}
    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(out)

    try:
        args = train_api.parse_train_args(
            [
                "supervised",
                "--dataset",
                "iris",
                "--model",
                "mlp",
            ]
        )
        assert args.mode == "supervised"
        assert args.model == "mlp"
        assert args.dataset == "iris"
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        datasets_registry.DATASET_REGISTRY._items = original_dataset_items
        plugins.reset_capsule_plugin_cache()


def test_fresh_capsule_examples_do_not_pollute_plugin_listings(tmp_path, monkeypatch, capsys) -> None:
    out = create.create_capsule_scaffold(
        capsule_name="quiet_capsule",
        base_dir=tmp_path,
        register=False,
    )
    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(out)
    try:
        rc = list_cli.main(["optimizers", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["sources"]["capsules"] == {}

        rc = list_cli.main(["models", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["sources"]["capsules"] == {}
    finally:
        plugins.reset_capsule_plugin_cache()
