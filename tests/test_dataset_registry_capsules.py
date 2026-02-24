from __future__ import annotations

import importlib
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
plugins = importlib.import_module("lelabo.core.utils.capsule_plugins")
datasets_registry = importlib.import_module("lelabo.supervised.datasets.registry")
capsule_registry = importlib.import_module("lelabo.capsule.registry")


def test_get_dataset_names_includes_installed_capsule_datasets(tmp_path, monkeypatch) -> None:
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

    caps_dir = tmp_path / "caps_store"
    capsule_registry.add_capsule_entry(
        capsule_id="capsule_ds_id",
        capsule_path=capsule_root,
        manifest={"kind": "config_only", "created_at": "2026-02-22T00:00:00Z", "source": {"path": str(capsule_root)}},
        alias="capsule_ds_alias",
        capsules_dir=caps_dir,
    )
    monkeypatch.setenv("LELABO_CAPSULES_DIR", str(caps_dir))

    original_dataset_items = dict(datasets_registry.DATASET_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        names = datasets_registry.get_dataset_names()
        assert "installed_capsule_dataset" in names
    finally:
        datasets_registry.DATASET_REGISTRY._items = original_dataset_items
        plugins.reset_capsule_plugin_cache()
