from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
callbacks_api = importlib.import_module("lelabo.callbacks")
callbacks_registry = importlib.import_module("lelabo.callbacks.registry")
core_callbacks = importlib.import_module("lelabo.core.callbacks")
plugins = importlib.import_module("lelabo.core.utils.capsule_plugins")
trainer_api = importlib.import_module("lelabo.core.trainer")
train_api = importlib.import_module("lelabo.api.train")


def test_builtin_callbacks_include_early_stopping() -> None:
    names = set(callbacks_api.get_callback_names())
    assert "earlystopping" in names


def test_build_early_stopping_from_registry_context() -> None:
    ctx = callbacks_api.CallbackContext(
        args=Namespace(),
        mode="supervised",
        dataset="iris",
        params={"monitor": "val.acc", "mode": "auto", "patience": 3},
    )
    callback = callbacks_api.build_callback("earlystopping", ctx)
    assert isinstance(callback, core_callbacks.EarlyStopping)
    assert callback.cfg.mode == "max"
    assert callback.cfg.patience == 3


def test_parse_train_args_exposes_callbacks_list() -> None:
    args = train_api.parse_train_args(["supervised", "--dataset", "iris"])
    assert isinstance(args.callbacks, list)
    assert args.callbacks
    assert args.callbacks[0]["name"] == "earlystopping"


def test_callback_registry_can_load_capsule_plugin(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_with_callbacks"
    (capsule_root / "callbacks").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_with_callbacks\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "callbacks" / "example.py").write_text(
        "from lelabo.callbacks import CallbackContext, register_callback\n\n"
        "@register_callback('capsule_callback')\n"
        "def build_capsule_callback(ctx: CallbackContext):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    original_items = dict(callbacks_registry.CALLBACK_REGISTRY._items)
    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(capsule_root)
    try:
        names = callbacks_api.get_callback_names()
        assert "capsule_callback" in names
    finally:
        callbacks_registry.CALLBACK_REGISTRY._items = original_items
        plugins.reset_capsule_plugin_cache()


def test_trainer_callback_hook_is_duck_typed() -> None:
    assert trainer_api.Trainer._call_callback_hook(object(), "on_epoch_end", None) is None


def test_early_stopping_integration_stops_iris_training_early(tmp_path) -> None:
    run_dir = tmp_path / "run_es_iris"
    cfg = tmp_path / "train.supervised.toml"
    cfg.write_text(
        "\n".join(
            [
                'task = "supervised"',
                "",
                "[dataset]",
                'name = "iris"',
                "",
                "[model.params]",
                "hidden = 32",
                "layers = 1",
                "",
                "[train]",
                "epochs = 12",
                "batch = 128",
                "val_frac = 0.1",
                "",
                "[runtime]",
                "verbose = 0",
                f'run_dir = "{run_dir.as_posix()}"',
                "",
                "[[callbacks]]",
                'name = "earlystopping"',
                "enabled = true",
                "[callbacks.params]",
                'monitor = "val.acc"',
                'mode = "max"',
                "patience = 0",
                "min_delta = 1.1",
                "warmup = 0",
                "restore_best = true",
            ]
        ),
        encoding="utf-8",
    )

    summary = train_api.run_train_from_argv(["supervised", "--config", str(cfg)])
    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists()

    train_epochs = 0
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("t") == "train":
            train_epochs += 1

    assert train_epochs == 2
    assert 1 <= int(summary["train"]["best_epoch_by_val"]) <= 2
