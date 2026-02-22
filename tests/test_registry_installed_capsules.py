from __future__ import annotations

import importlib
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
plugins = importlib.import_module("lab.core.utils.capsule_plugins")
capsule_registry = importlib.import_module("lab.capsule.registry")
models_registry = importlib.import_module("lab.models.registry")
rules_registry = importlib.import_module("lab.update_rules.registry")
metrics_registry = importlib.import_module("lab.metrics.registry")


def test_get_model_names_includes_installed_capsules(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_models"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "models" / "m.py").write_text(
        "from lab.models.registry import register_model\n\n"
        "@register_model('installed_capsule_model')\n"
        "def build_installed_capsule_model(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    caps_dir = tmp_path / "caps_store"
    capsule_registry.add_capsule_entry(
        capsule_id="caps_model_id",
        capsule_path=capsule_root,
        manifest={"kind": "config_only", "created_at": "2026-02-22T00:00:00Z", "source": {"path": str(capsule_root)}},
        alias="caps_model_alias",
        capsules_dir=caps_dir,
    )

    original_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        names = models_registry.get_model_names(capsules_dir=caps_dir)
        assert "installed_capsule_model" in names
    finally:
        models_registry.MODEL_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()


def test_get_update_rule_names_includes_installed_capsules(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_rules"
    (capsule_root / "update_rules").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "update_rules" / "r.py").write_text(
        "from lab.update_rules.registry import register_update_rule\n\n"
        "@register_update_rule('installed_capsule_rule')\n"
        "def build_installed_capsule_rule(ctx):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    caps_dir = tmp_path / "caps_store"
    capsule_registry.add_capsule_entry(
        capsule_id="caps_rule_id",
        capsule_path=capsule_root,
        manifest={"kind": "config_only", "created_at": "2026-02-22T00:00:00Z", "source": {"path": str(capsule_root)}},
        alias="caps_rule_alias",
        capsules_dir=caps_dir,
    )

    original_items = dict(rules_registry.UPDATE_RULE_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        names = rules_registry.get_update_rule_names(capsules_dir=caps_dir)
        assert "installed_capsule_rule" in names
    finally:
        rules_registry.UPDATE_RULE_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()


def test_get_metric_names_includes_installed_capsules(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_metrics"
    (capsule_root / "metrics").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "metrics" / "metric.py").write_text(
        "from lab.metrics.registry import register_metric\n\n"
        "@register_metric('installed_capsule_metric')\n"
        "def build_installed_capsule_metric(ctx):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    caps_dir = tmp_path / "caps_store"
    capsule_registry.add_capsule_entry(
        capsule_id="caps_metric_id",
        capsule_path=capsule_root,
        manifest={"kind": "config_only", "created_at": "2026-02-22T00:00:00Z", "source": {"path": str(capsule_root)}},
        alias="caps_metric_alias",
        capsules_dir=caps_dir,
    )

    original_items = dict(metrics_registry.METRIC_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        names = metrics_registry.get_metric_names(capsules_dir=caps_dir)
        assert "installed_capsule_metric" in names
    finally:
        metrics_registry.METRIC_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()
