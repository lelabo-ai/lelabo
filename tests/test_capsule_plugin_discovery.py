from __future__ import annotations

import importlib
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
plugins = importlib.import_module("lab.core.utils.capsule_plugins")
models_registry = importlib.import_module("lab.models.registry")
rules_registry = importlib.import_module("lab.update_rules.registry")


def test_capsule_plugins_are_loaded_into_registries(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_a"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "update_rules").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_a\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )

    (capsule_root / "models" / "capsule_model.py").write_text(
        "from lab.models.registry import register_model\n"
        "import torch.nn as nn\n\n"
        "@register_model('capsule_identity')\n"
        "def build_capsule_identity(ctx, args):\n"
        "    return nn.Identity()\n",
        encoding="utf-8",
    )
    (capsule_root / "update_rules" / "capsule_rule.py").write_text(
        "from lab.update_rules.registry import register_update_rule\n\n"
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
        "from lab.models.registry import register_model\n\n"
        f"@register_model('{model_a}')\n"
        "def build_a(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (cap_b / "models" / "b.py").write_text(
        "from lab.models.registry import register_model\n\n"
        f"@register_model('{model_b}')\n"
        "def build_b(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    monkeypatch.delenv("LELABO_CAPSULES_DIR", raising=False)
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
