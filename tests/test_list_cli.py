from __future__ import annotations

import importlib
import json
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
list_cli = importlib.import_module("lelabo.cli.commands.list")
plugins = importlib.import_module("lelabo.capsule.plugins")
models_registry = importlib.import_module("lelabo.models.registry")
optimizers_registry = importlib.import_module("lelabo.optimizers.registry")
schedulers_registry = importlib.import_module("lelabo.schedulers.registry")
callbacks_registry = importlib.import_module("lelabo.callbacks.registry")
datasets_registry = importlib.import_module("lelabo.supervised.datasets.registry")
capsule_registry = importlib.import_module("lelabo.capsule.registry")


def test_list_cli_all_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        list_cli,
        "_collect",
        lambda *args, **kwargs: {
            "update_rules": ["bp", "fa"],
            "datasets": ["iris"],
            "models": ["cnn", "mlp"],
            "initializers": ["none", "kaiming_uniform"],
            "optimizers": ["adamw"],
            "losses": ["cross_entropy", "mse"],
            "metrics": ["acc"],
            "schedulers": ["step"],
            "callbacks": ["earlystopping"],
        },
    )

    rc = list_cli.main(["all"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "update_rules (2)" in out
    assert "- bp" in out
    assert "datasets (1)" in out
    assert "models (2)" in out
    assert "initializers (2)" in out
    assert "optimizers (1)" in out
    assert "losses (2)" in out
    assert "metrics (1)" in out
    assert "schedulers (1)" in out
    assert "callbacks (1)" in out


def test_list_cli_single_target_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(list_cli, "get_update_rule_names", lambda **kwargs: ["dfa", "bp"])

    rc = list_cli.main(["update-rules", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert sorted(payload.keys()) == ["update_rules"]
    assert payload["update_rules"] == ["bp", "dfa"]


def test_list_cli_algos_alias_maps_to_update_rules(monkeypatch, capsys) -> None:
    monkeypatch.setattr(list_cli, "get_update_rule_names", lambda **kwargs: ["dfa", "bp"])

    rc = list_cli.main(["algos", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert sorted(payload.keys()) == ["update_rules"]
    assert payload["update_rules"] == ["bp", "dfa"]


def test_list_cli_unknown_target_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        list_cli.main(["unknown_target"])


def test_list_cli_schedulers_target_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(list_cli, "get_scheduler_names", lambda **kwargs: ["cosine", "step"])

    rc = list_cli.main(["schedulers", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert sorted(payload.keys()) == ["schedulers"]
    assert payload["schedulers"] == ["cosine", "step"]


def test_list_cli_optimizers_target_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(list_cli, "get_optimizer_names", lambda **kwargs: ["adamw", "sgd"])

    rc = list_cli.main(["optimizers", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert sorted(payload.keys()) == ["optimizers"]
    assert payload["optimizers"] == ["adamw", "sgd"]


def test_list_cli_losses_target_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(list_cli, "get_loss_names", lambda **kwargs: ["cross_entropy", "bce_with_logits"])

    rc = list_cli.main(["losses", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert sorted(payload.keys()) == ["losses"]
    assert payload["losses"] == ["bce_with_logits", "cross_entropy"]


def test_list_cli_initializers_target_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(list_cli, "get_initializer_names", lambda **kwargs: ["none", "xavier_uniform"])

    rc = list_cli.main(["initializers", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert sorted(payload.keys()) == ["initializers"]
    assert payload["initializers"] == ["none", "xavier_uniform"]


def test_list_cli_callbacks_target_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(list_cli, "get_callback_names", lambda **kwargs: ["earlystopping", "my_cb"])

    rc = list_cli.main(["callbacks", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert sorted(payload.keys()) == ["callbacks"]
    assert payload["callbacks"] == ["earlystopping", "my_cb"]


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

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["models", "--capsule", str(capsule_root), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "capsule_list_model" in payload["models"]
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
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

    original_dataset_items = dict(datasets_registry.DATASET_REGISTRY._items)
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
        assert "capsule_list_dataset" in payload["datasets"]
    finally:
        datasets_registry.DATASET_REGISTRY._items = original_dataset_items
        plugins.reset_capsule_plugin_cache()


def test_list_cli_auto_includes_installed_capsules(tmp_path, monkeypatch, capsys) -> None:
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

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["models", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "auto_capsule_model" in payload["models"]
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        plugins.reset_capsule_plugin_cache()


def test_list_cli_auto_includes_installed_capsules_from_capsules_dir_flag(tmp_path, monkeypatch, capsys) -> None:
    capsule_root = tmp_path / "capsule_installed_models_flag"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "models" / "auto_model_flag.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        "@register_model('auto_capsule_model_flag')\n"
        "def build_auto_capsule_model_flag(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (capsule_root / "manifest.json").write_text("{}", encoding="utf-8")

    caps_dir = tmp_path / "caps_store_flag"
    capsule_registry.add_capsule_entry(
        capsule_id="auto_caps_id_flag",
        capsule_path=capsule_root,
        manifest={"kind": "config_only", "created_at": "2026-02-22T00:00:00Z", "source": {"path": str(capsule_root)}},
        alias="auto_caps_alias_flag",
        capsules_dir=caps_dir,
    )
    monkeypatch.delenv("LELABO_CAPSULES_DIR", raising=False)

    original_model_items = dict(models_registry.MODEL_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["models", "--capsules-dir", str(caps_dir), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "auto_capsule_model_flag" in payload["models"]
    finally:
        models_registry.MODEL_REGISTRY._items = original_model_items
        plugins.reset_capsule_plugin_cache()


def test_list_cli_can_include_capsule_schedulers_from_path(tmp_path, capsys) -> None:
    capsule_root = tmp_path / "capsule_schedulers"
    (capsule_root / "schedulers").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_schedulers\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "schedulers" / "extra_scheduler.py").write_text(
        "from lelabo.schedulers import register_scheduler\n\n"
        "@register_scheduler('capsule_list_scheduler')\n"
        "def build_capsule_list_scheduler(ctx):\n"
        "    return None\n",
        encoding="utf-8",
    )

    original_scheduler_items = dict(schedulers_registry.SCHEDULER_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["schedulers", "--capsule", str(capsule_root), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "capsule_list_scheduler" in payload["schedulers"]
    finally:
        schedulers_registry.SCHEDULER_REGISTRY._items = original_scheduler_items
        plugins.reset_capsule_plugin_cache()


def test_list_cli_can_include_capsule_optimizers_from_path(tmp_path, capsys) -> None:
    capsule_root = tmp_path / "capsule_optimizers"
    (capsule_root / "optimizers").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_optimizers\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "optimizers" / "extra_optimizer.py").write_text(
        "import torch\n"
        "from lelabo.optimizers import register_optimizer\n\n"
        "@register_optimizer('capsule_list_optimizer')\n"
        "def build_capsule_list_optimizer(ctx):\n"
        "    return torch.optim.AdamW(ctx.params, lr=ctx.lr, weight_decay=ctx.weight_decay)\n",
        encoding="utf-8",
    )

    original_optimizer_items = dict(optimizers_registry.OPTIMIZER_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["optimizers", "--capsule", str(capsule_root), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "capsule_list_optimizer" in payload["optimizers"]
    finally:
        optimizers_registry.OPTIMIZER_REGISTRY._items = original_optimizer_items
        plugins.reset_capsule_plugin_cache()


def test_list_cli_can_include_capsule_callbacks_from_path(tmp_path, capsys) -> None:
    capsule_root = tmp_path / "capsule_callbacks"
    (capsule_root / "callbacks").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_callbacks\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "callbacks" / "extra_callback.py").write_text(
        "from lelabo.callbacks import register_callback\n\n"
        "@register_callback('capsule_list_callback')\n"
        "def build_capsule_list_callback(ctx):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    original_callback_items = dict(callbacks_registry.CALLBACK_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    try:
        rc = list_cli.main(["callbacks", "--capsule", str(capsule_root), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "capsule_list_callback" in payload["callbacks"]
    finally:
        callbacks_registry.CALLBACK_REGISTRY._items = original_callback_items
        plugins.reset_capsule_plugin_cache()
