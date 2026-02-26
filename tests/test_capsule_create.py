from __future__ import annotations

import importlib
import json
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
create = importlib.import_module("lelabo.capsule.create")
registry = importlib.import_module("lelabo.capsule.registry")
plugins = importlib.import_module("lelabo.core.utils.capsule_plugins")
models_registry = importlib.import_module("lelabo.models.registry")
datasets_registry = importlib.import_module("lelabo.supervised.datasets.registry")
train_api = importlib.import_module("lelabo.api.train")
lab_pkg = importlib.import_module("lelabo")


def test_create_capsule_scaffold_creates_expected_layout(tmp_path) -> None:
    capsules_dir = tmp_path / ".lelabo" / "capsules"
    out = create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=tmp_path,
        capsules_dir=capsules_dir,
    )

    assert out == (tmp_path / "demo_capsule")
    assert out.is_dir()
    assert (out / "models").is_dir()
    assert (out / "update_rules").is_dir()
    assert (out / "datasets").is_dir()
    assert (out / "metrics").is_dir()
    assert (out / "optimizers").is_dir()
    assert (out / "schedulers").is_dir()
    assert (out / "callbacks").is_dir()
    assert (out / "configs").is_dir()
    assert (out / "runs").is_dir()
    assert (out / "README.md").exists()
    assert (out / "capsule.toml").exists()
    assert (out / "manifest.json").exists()
    assert (out / "models" / "__init__.py").exists()
    assert (out / "models" / "example.py").exists()
    assert (out / "update_rules" / "example.py").exists()
    assert (out / "datasets" / "example.py").exists()
    assert (out / "metrics" / "example.py").exists()
    assert (out / "optimizers" / "example.py").exists()
    assert (out / "schedulers" / "example.py").exists()
    assert (out / "callbacks" / "example.py").exists()
    assert (out / "configs" / "README.md").exists()
    assert (out / "configs" / "train.supervised.quickstart.toml").exists()
    assert (out / "configs" / "train.supervised.detailed.toml").exists()
    assert (out / "configs" / "train.rl.detailed.toml").exists()
    assert (out / "runs" / "example.py").exists()
    model_example = (out / "models" / "example.py").read_text(encoding="utf-8")
    metric_example = (out / "metrics" / "example.py").read_text(encoding="utf-8")
    readme_text = (out / "README.md").read_text(encoding="utf-8")
    capsule_toml = (out / "capsule.toml").read_text(encoding="utf-8")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    configs_readme = (out / "configs" / "README.md").read_text(encoding="utf-8")
    cfg_quick = (out / "configs" / "train.supervised.quickstart.toml").read_text(encoding="utf-8")
    assert "register_model" in model_example
    assert "register_metric" in metric_example
    assert "register_metric_fn" in metric_example
    assert "ClassificationMetricBase" in metric_example
    assert "register_optimizer" in readme_text
    assert "register_callback" in readme_text
    assert "lelabo_version" in capsule_toml
    assert str(manifest.get("lelabo_version", "")).strip()
    assert 'config_version = "1.0"' in cfg_quick
    assert f'lelabo_version = "{lab_pkg.__version__}"' in cfg_quick
    assert 'config_version = "auto"' not in cfg_quick
    assert "layered config resolution" in configs_readme.lower()
    assert "train.supervised.quickstart.toml" in configs_readme
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
