from __future__ import annotations

import importlib
import json
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
list_cli = importlib.import_module("lelabo.cli.commands.list")
plugins = importlib.import_module("lelabo.capsule.plugins")
capsule_registry = importlib.import_module("lelabo.capsule.registry")


def test_list_cli_all_text_groups_builtins_and_capsule(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        list_cli,
        "_snapshot_rows",
        lambda *args, **kwargs: {
            "models": {"builtins": ["cnn", "mlp"], "capsule": ["paper_model"]},
            "optimizers": {"builtins": ["adamw"], "capsule": []},
        },
    )

    rc = list_cli.main(["all"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "models:" in out
    assert "builtins:" in out
    assert "- cnn" in out
    assert "capsule:" in out
    assert "- paper_model" in out
    assert "optimizers:" in out


def test_list_cli_single_target_json_is_grouped(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        list_cli,
        "_snapshot_rows",
        lambda *args, **kwargs: {"update_rules": {"builtins": ["bp"], "capsule": ["dfa"]}},
    )

    rc = list_cli.main(["update-rules", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": "lelabo.cli.list/v1",
        "target": "update_rules",
        "sources": {"builtins": ["bp"], "capsule": ["dfa"]},
    }


def test_list_cli_all_json_uses_registries_wrapper(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        list_cli,
        "_snapshot_rows",
        lambda *args, **kwargs: {"models": {"builtins": ["cnn"], "capsule": ["paper_model"]}},
    )

    rc = list_cli.main(["all", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": "lelabo.cli.list/v1",
        "target": "all",
        "registries": {"models": {"builtins": ["cnn"], "capsule": ["paper_model"]}},
    }


def test_list_cli_algos_alias_maps_to_update_rules(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        list_cli,
        "_snapshot_rows",
        lambda *args, **kwargs: {"update_rules": {"builtins": ["bp"], "capsule": ["dfa"]}},
    )

    rc = list_cli.main(["algos", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "update_rules"


def test_list_cli_unknown_target_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        list_cli.main(["unknown_target"])


def test_list_cli_can_include_capsule_models_from_path(tmp_path, capsys) -> None:
    capsule_root = tmp_path / "capsule_models"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_models\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "models" / "extra_model.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('capsule_list_model')\n"
        "def build_capsule_list_model(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["models", "--capsule", str(capsule_root), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "capsule_list_model" in payload["sources"]["capsule"]
    finally:
        plugins.reset_capsule_plugin_cache()


def test_list_cli_can_include_capsule_datasets_from_alias(tmp_path, capsys) -> None:
    capsule_root = tmp_path / "capsule_datasets"
    (capsule_root / "datasets").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_datasets\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")
    (capsule_root / "datasets" / "extra_dataset.py").write_text(
        "from lelabo.supervised.datasets.registry import register_dataset\n\n"
        "@register_dataset('capsule_list_dataset')\n"
        "def build_capsule_list_dataset(**kwargs):\n"
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

    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(
            [
                "datasets",
                "--capsule",
                "capsule_ds_alias",
                "--capsules-dir",
                str(caps_dir),
                "--json",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "capsule_list_dataset" in payload["sources"]["capsule"]
    finally:
        plugins.reset_capsule_plugin_cache()


def test_list_cli_uses_configured_store_dir_for_capsule_alias_resolution(tmp_path, monkeypatch, capsys) -> None:
    capsule_root = tmp_path / "capsule_cfg_store"
    (capsule_root / "datasets").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_cfg_store\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "datasets" / "cfg_dataset.py").write_text(
        "from lelabo.supervised.datasets.registry import register_dataset\n\n"
        "@register_dataset('cfg_store_dataset')\n"
        "def build_cfg_store_dataset(**kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")

    caps_dir = tmp_path / "configured_caps_store"
    capsule_registry.add_capsule_entry(
        capsule_id="cfg_caps_id",
        capsule_path=capsule_root,
        manifest={"kind": "config_only", "created_at": "2026-02-22T00:00:00Z", "source": {"path": str(capsule_root)}},
        alias="cfg_caps_alias",
        capsules_dir=caps_dir,
    )

    monkeypatch.setattr(
        list_cli,
        "load_effective_settings",
        lambda: {"capsules": {"store_dir": str(caps_dir)}},
    )

    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["datasets", "--capsule", "cfg_caps_alias", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "cfg_store_dataset" in payload["sources"]["capsule"]
    finally:
        plugins.reset_capsule_plugin_cache()


def test_list_cli_does_not_auto_include_stored_capsules(tmp_path, monkeypatch, capsys) -> None:
    capsule_root = tmp_path / "capsule_installed_models"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "models" / "auto_model.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('auto_capsule_model')\n"
        "def build_auto_capsule_model(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")

    caps_dir = tmp_path / "caps_store_auto"
    capsule_registry.add_capsule_entry(
        capsule_id="auto_caps_id",
        capsule_path=capsule_root,
        manifest={"kind": "config_only", "created_at": "2026-02-22T00:00:00Z", "source": {"path": str(capsule_root)}},
        alias="auto_caps_alias",
        capsules_dir=caps_dir,
    )
    monkeypatch.setenv("LELABO_CAPSULES_DIR", str(caps_dir))

    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["models", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "auto_capsule_model" not in payload["sources"]["capsule"]
    finally:
        plugins.reset_capsule_plugin_cache()


def test_list_cli_auto_includes_workspace_capsules(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_list_runtime"
    workspace.mkdir()
    capsule_root = workspace / "capsule_models"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_models\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "models" / "workspace_model.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('workspace_capsule_model')\n"
        "def build_workspace_capsule_model(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)

    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["models", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "workspace_capsule_model" in payload["sources"]["capsule"]
    finally:
        plugins.reset_capsule_plugin_cache()
