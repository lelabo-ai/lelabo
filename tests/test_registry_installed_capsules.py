from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
plugins = importlib.import_module("lelabo.capsule.plugins")
capsule_registry = importlib.import_module("lelabo.capsule.registry")
models_registry = importlib.import_module("lelabo.models.registry")
rules_registry = importlib.import_module("lelabo.update_rules.registry")
metrics_registry = importlib.import_module("lelabo.metrics.registry")


def test_get_model_names_can_include_explicit_capsule_roots(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_models"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "models" / "m.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('installed_capsule_model')\n"
        "def build_installed_capsule_model(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    original_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        names = models_registry.get_model_names(extra_capsule_roots=[capsule_root])
        assert "installed_capsule_model" in names
    finally:
        models_registry.MODEL_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()


def test_get_update_rule_names_can_include_explicit_capsule_roots(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_rules"
    (capsule_root / "update_rules").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "update_rules" / "r.py").write_text(
        "from lelabo.update_rules.registry import register_update_rule\n\n"
        "@register_update_rule('installed_capsule_rule')\n"
        "def build_installed_capsule_rule(ctx):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    original_items = dict(rules_registry.UPDATE_RULE_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        names = rules_registry.get_update_rule_names(extra_capsule_roots=[capsule_root])
        assert "installed_capsule_rule" in names
    finally:
        rules_registry.UPDATE_RULE_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()


def test_get_metric_names_can_include_explicit_capsule_roots(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_metrics"
    (capsule_root / "metrics").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "metrics" / "metric.py").write_text(
        "from lelabo.metrics.registry import register_metric\n\n"
        "@register_metric('installed_capsule_metric')\n"
        "def build_installed_capsule_metric(ctx):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    original_items = dict(metrics_registry.METRIC_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        names = metrics_registry.get_metric_names(extra_capsule_roots=[capsule_root])
        assert "installed_capsule_metric" in names
    finally:
        metrics_registry.METRIC_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()


def test_broken_explicit_capsule_root_raises_runtime_error(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_broken"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "models" / "broken.py").write_text(
        "raise RuntimeError('boom from broken capsule')\n",
        encoding="utf-8",
    )

    original_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        with pytest.raises(RuntimeError):
            models_registry.get_model_names(extra_capsule_roots=[capsule_root])
    finally:
        models_registry.MODEL_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()


def test_explicit_capsule_root_missing_external_dependency_warns_and_skips(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_missing_dep"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "models" / "missing_dep.py").write_text(
        "import definitely_missing_installed_capsule_dep\n",
        encoding="utf-8",
    )

    original_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        with pytest.warns(RuntimeWarning, match="missing_dependency='definitely_missing_installed_capsule_dep'"):
            names = models_registry.get_model_names(extra_capsule_roots=[capsule_root])
        assert "mlp" in names
    finally:
        models_registry.MODEL_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()
