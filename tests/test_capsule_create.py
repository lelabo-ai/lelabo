from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
create = importlib.import_module("lab.capsule.create")
registry = importlib.import_module("lab.capsule.registry")
plugins = importlib.import_module("lab.core.utils.capsule_plugins")
models_registry = importlib.import_module("lab.models.registry")
datasets_registry = importlib.import_module("lab.supervised.datasets.registry")
train_api = importlib.import_module("lab.api.train")


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
    assert (out / "configs" / "example.py").exists()
    assert (out / "runs" / "example.py").exists()
    model_example = (out / "models" / "example.py").read_text(encoding="utf-8")
    assert "register_model" in model_example
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
                "example_mnist",
                "--model",
                "example_mlp",
            ]
        )
        assert args.mode == "supervised"
        assert args.model == "example_mlp"
        assert args.dataset == "example_mnist"
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        datasets_registry.DATASET_REGISTRY._items = original_dataset_items
        plugins.reset_capsule_plugin_cache()
