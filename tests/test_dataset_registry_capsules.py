from __future__ import annotations

import importlib
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
plugins = importlib.import_module("lelabo.capsule.plugins")
datasets_registry = importlib.import_module("lelabo.supervised.datasets.registry")
capsule_registry = importlib.import_module("lelabo.capsule.registry")


def test_get_dataset_names_can_include_explicit_capsule_datasets(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_ds"
    (capsule_root / "datasets").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "datasets" / "extra_dataset.py").write_text(
        "from lelabo.supervised.datasets.registry import register_dataset\n\n"
        "@register_dataset('installed_capsule_dataset')\n"
        "def build_installed_capsule_dataset(**kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )

    original_dataset_items = dict(datasets_registry.DATASET_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        names = datasets_registry.get_dataset_names(extra_capsule_roots=[capsule_root])
        assert "installed_capsule_dataset" in names
    finally:
        datasets_registry.DATASET_REGISTRY._items = original_dataset_items
        plugins.reset_capsule_plugin_cache()
