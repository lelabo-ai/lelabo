from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
plugins = importlib.import_module("lelabo.capsule.plugins")
models_registry = importlib.import_module("lelabo.models.registry")
rules_registry = importlib.import_module("lelabo.update_rules.registry")


def test_capsule_plugins_are_loaded_into_registries(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_a"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "update_rules").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_a\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )

    (capsule_root / "models" / "capsule_model.py").write_text(
        "from lelabo.models.registry import register_model\n"
        "import torch.nn as nn\n\n"
        "@register_model('capsule_identity')\n"
        "def build_capsule_identity(ctx, args):\n"
        "    return nn.Identity()\n",
        encoding="utf-8",
    )
    (capsule_root / "update_rules" / "capsule_rule.py").write_text(
        "from lelabo.update_rules.registry import register_update_rule\n\n"
        "@register_update_rule('capsule_dummy_rule')\n"
        "def build_capsule_dummy_rule(ctx):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    original_rule_items = dict(rules_registry.UPDATE_RULE_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(capsule_root)

    try:
        model_names = models_registry.get_model_names()
        rule_names = rules_registry.get_update_rule_names()
        assert "capsule_identity" in model_names
        assert "capsule_dummy_rule" in rule_names
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        rules_registry.UPDATE_RULE_REGISTRY._items = original_rule_items
        plugins.reset_capsule_plugin_cache()


def test_active_capsule_registry_state_is_isolated_between_roots(tmp_path, monkeypatch) -> None:
    cap_a = tmp_path / "capsule_a"
    cap_b = tmp_path / "capsule_b"

    (cap_a / "models").mkdir(parents=True)
    (cap_b / "models").mkdir(parents=True)
    (cap_a / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_a\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (cap_b / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_b\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )

    model_a = "isolated_model_caps_a"
    model_b = "isolated_model_caps_b"
    (cap_a / "models" / "a.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        f"@register_model('{model_a}')\n"
        "def build_a(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (cap_b / "models" / "b.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        f"@register_model('{model_b}')\n"
        "def build_b(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    empty_capsules_dir = tmp_path / "empty_capsules_dir"
    empty_capsules_dir.mkdir()
    monkeypatch.setenv("LELABO_CAPSULES_DIR", str(empty_capsules_dir))
    try:
        monkeypatch.chdir(cap_a)
        names_a = set(models_registry.get_model_names())
        assert model_a in names_a
        assert model_b not in names_a

        monkeypatch.chdir(cap_b)
        names_b = set(models_registry.get_model_names())
        assert model_b in names_b
        assert model_a not in names_b
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        plugins.reset_capsule_plugin_cache()


def test_capsule_plugin_missing_external_dependency_warns_and_skips(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_missing_dep"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_missing_dep\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    plugin_path = capsule_root / "models" / "missing_dep.py"
    plugin_path.write_text(
        "import definitely_missing_capsule_dep\n",
        encoding="utf-8",
    )

    plugins.reset_capsule_plugin_cache()
    try:
        with pytest.warns(RuntimeWarning, match="missing_dependency='definitely_missing_capsule_dep'"):
            loaded = plugins.load_capsule_plugins(kinds=("models",), capsule_root=capsule_root)
        assert loaded == []
    finally:
        plugins.reset_capsule_plugin_cache()


def test_capsule_plugin_internal_import_error_raises_with_context(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_internal_bug"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_internal_bug\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    plugin_path = capsule_root / "models" / "broken.py"
    plugin_path.write_text(
        "from lelabo.models.registry import missing_symbol\n",
        encoding="utf-8",
    )

    plugins.reset_capsule_plugin_cache()
    try:
        with pytest.raises(RuntimeError, match="kind='models'.*capsule_internal_bug"):
            plugins.load_capsule_plugins(kinds=("models",), capsule_root=capsule_root)
    finally:
        plugins.reset_capsule_plugin_cache()


def test_capsule_plugin_missing_register_import_has_actionable_error(tmp_path) -> None:
    capsule_root = tmp_path / "capsule_missing_register_import"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_missing_register_import\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    plugin_path = capsule_root / "models" / "broken.py"
    plugin_path.write_text(
        "@register_model('broken_template_model')\n"
        "def build_broken(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    plugins.reset_capsule_plugin_cache()
    try:
        with pytest.raises(RuntimeError, match="does not import 'register_model'"):
            plugins.load_capsule_plugins(kinds=("models",), capsule_root=capsule_root)
    finally:
        plugins.reset_capsule_plugin_cache()


def test_active_capsule_helper_edit_invalidates_plugin_index(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_helper_refresh"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_helper_refresh\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "helpers.py").write_text('MODEL_NAME = "helper_model_v1"\n', encoding="utf-8")
    (capsule_root / "models" / "plugin.py").write_text(
        "from helpers import MODEL_NAME\n"
        "from lelabo.models.registry import register_model\n\n"
        "@register_model(MODEL_NAME)\n"
        "def build_helper_model(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(capsule_root)
    try:
        names_v1 = set(models_registry.get_model_names())
        assert "helper_model_v1" in names_v1
        assert "helper_model_v2" not in names_v1

        (capsule_root / "helpers.py").write_text('MODEL_NAME = "helper_model_v2"\n', encoding="utf-8")

        names_v2 = set(models_registry.get_model_names())
        assert "helper_model_v2" in names_v2
        assert "helper_model_v1" not in names_v2
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        plugins.reset_capsule_plugin_cache()


def test_active_capsule_supports_multiple_plugins_in_one_file(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_multi_models"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_multi_models\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "models" / "multi.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('capsule_model_alpha')\n"
        "def build_alpha(ctx, args):\n"
        "    return None\n\n"
        "@register_model('capsule_model_beta')\n"
        "def build_beta(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(capsule_root)
    try:
        names = set(models_registry.get_model_names())
        assert "capsule_model_alpha" in names
        assert "capsule_model_beta" in names
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        plugins.reset_capsule_plugin_cache()


def test_listing_capsule_models_does_not_import_plugin_modules_into_main_process(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_no_main_import"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_no_main_import\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    plugin_file = (capsule_root / "models" / "probe.py").resolve()
    plugin_file.write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('capsule_probe_model')\n"
        "def build_probe(ctx, args):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(capsule_root)
    try:
        names = set(models_registry.get_model_names())
        assert "capsule_probe_model" in names
        imported_files = {
            str(Path(path).resolve())
            for mod in list(sys.modules.values())
            for path in [getattr(mod, "__file__", None)]
            if isinstance(path, str) and path.strip()
        }
        assert str(plugin_file) not in imported_files
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        plugins.reset_capsule_plugin_cache()


def test_build_model_lazily_imports_only_selected_capsule_plugin(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_lazy_build"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_lazy_build\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    selected_file = (capsule_root / "models" / "selected.py").resolve()
    skipped_file = (capsule_root / "models" / "skipped.py").resolve()
    selected_file.write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('selected_capsule_model')\n"
        "def build_selected(ctx, args):\n"
        "    return {'name': 'selected'}\n",
        encoding="utf-8",
    )
    skipped_file.write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('skipped_capsule_model')\n"
        "def build_skipped(ctx, args):\n"
        "    return {'name': 'skipped'}\n",
        encoding="utf-8",
    )

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(capsule_root)
    try:
        ctx = models_registry.ModelContext(dataset="iris", num_classes=3, in_dim=4)
        out = models_registry.build_model("selected_capsule_model", ctx, args=None)
        assert out == {"name": "selected"}

        imported_files = {
            str(Path(path).resolve())
            for mod in list(sys.modules.values())
            for path in [getattr(mod, "__file__", None)]
            if isinstance(path, str) and path.strip()
        }
        assert str(selected_file) in imported_files
        assert str(skipped_file) not in imported_files
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        plugins.reset_capsule_plugin_cache()
