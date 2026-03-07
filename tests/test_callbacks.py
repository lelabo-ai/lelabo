from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
callbacks_api = importlib.import_module("lelabo.callbacks")
callbacks_registry = importlib.import_module("lelabo.callbacks.registry")
core_callbacks = importlib.import_module("lelabo.core.callbacks")
plugins = importlib.import_module("lelabo.core.utils.capsule_plugins")
trainer_api = importlib.import_module("lelabo.core.trainer")
train_api = importlib.import_module("lelabo.api.train")
datasets_base = importlib.import_module("lelabo.supervised.datasets.base")


def test_builtin_callbacks_include_early_stopping() -> None:
    names = set(callbacks_api.get_callback_names())
    assert "earlystopping" in names


def test_build_early_stopping_from_registry_context() -> None:
    ctx = callbacks_api.CallbackContext(
        args=Namespace(),
        mode="supervised",
        dataset="iris",
        params={"monitor": "val.acc", "mode": "auto", "patience": 3, "restore_learner_state": True},
    )
    callback = callbacks_api.build_callback("earlystopping", ctx)
    assert isinstance(callback, core_callbacks.EarlyStopping)
    assert callback.cfg.mode == "max"
    assert callback.cfg.patience == 3
    assert callback.cfg.restore_learner_state is True


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


def test_early_stopping_capture_restore_learner_state_opt_in() -> None:
    class _DummyLearner:
        def __init__(self):
            self.marker = 3
            param = torch.nn.Parameter(torch.zeros(()), requires_grad=True)
            self.optimizer = torch.optim.SGD([param], lr=1e-2)

        def state_dict(self):
            return {"marker": int(self.marker)}

        def load_state_dict(self, state):
            self.marker = int(state["marker"])

    trainer = SimpleNamespace(
        model=torch.nn.Linear(2, 2),
        learner=_DummyLearner(),
        schedulers=[],
        state=SimpleNamespace(epoch=1, batch_idx=2, global_step=3, step=4),
    )

    cfg_disabled = core_callbacks.EarlyStoppingConfig(restore_learner_state=False)
    payload_disabled = core_callbacks.EarlyStopping._capture_state(trainer, cfg_disabled)
    assert "learner" not in payload_disabled

    cfg_enabled = core_callbacks.EarlyStoppingConfig(restore_learner_state=True)
    payload_enabled = core_callbacks.EarlyStopping._capture_state(trainer, cfg_enabled)
    assert "learner" in payload_enabled

    trainer.learner.marker = 99
    core_callbacks.EarlyStopping._restore_state(trainer, payload_enabled)
    assert trainer.learner.marker == 3


def test_early_stopping_integration_stops_iris_training_early(tmp_path, monkeypatch) -> None:
    iris_mod = importlib.import_module("lelabo.supervised.datasets.tabular.iris")

    def _make_tiny_iris_dataset(
        *,
        batch_size: int = 32,
        seed: int = 42,
        **_: object,
    ):
        g = torch.Generator().manual_seed(int(seed))
        x = torch.randn(60, 4, generator=g)
        y = torch.randint(0, 3, (60,), generator=g)

        x_tr, y_tr = x[:40], y[:40]
        x_va, y_va = x[40:50], y[40:50]
        x_te, y_te = x[50:], y[50:]

        train_loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=min(int(batch_size), int(x_tr.size(0))),
            shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(x_va, y_va),
            batch_size=min(int(batch_size), int(x_va.size(0))),
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(x_te, y_te),
            batch_size=min(int(batch_size), int(x_te.size(0))),
            shuffle=False,
        )
        return datasets_base.DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=3,
            in_dim=4,
            input_shape=None,
            x_test=x_te,
            y_test=y_te,
        )

    monkeypatch.setattr(iris_mod, "make_iris_dataset", _make_tiny_iris_dataset)

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
                'display = "none"',
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
