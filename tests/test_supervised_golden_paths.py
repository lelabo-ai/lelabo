from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
create_api = importlib.import_module("lelabo.capsule.create")
plugins = importlib.import_module("lelabo.capsule.plugins")
list_cli = importlib.import_module("lelabo.cli.commands.list")
train_api = importlib.import_module("lelabo.cli.commands.train")
datasets_base = importlib.import_module("lelabo.supervised.datasets.base")
runner_api = importlib.import_module("lelabo.supervised.runner")


def _tabular_bundle(*, batch_size: int, seed: int, in_dim: int = 4, num_classes: int = 3):
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(72, in_dim, generator=g)
    y = torch.randint(0, num_classes, (72,), generator=g)
    x_tr, y_tr = x[:48], y[:48]
    x_va, y_va = x[48:60], y[48:60]
    x_te, y_te = x[60:], y[60:]
    return datasets_base.DataBundle(
        train_loader=DataLoader(TensorDataset(x_tr, y_tr), batch_size=min(batch_size, len(x_tr)), shuffle=False),
        val_loader=DataLoader(TensorDataset(x_va, y_va), batch_size=min(batch_size, len(x_va)), shuffle=False),
        test_loader=DataLoader(TensorDataset(x_te, y_te), batch_size=min(batch_size, len(x_te)), shuffle=False),
        num_classes=num_classes,
        in_dim=in_dim,
        input_shape=None,
        x_test=x_te,
        y_test=y_te,
    )


def _image_bundle(
    *,
    batch_size: int,
    seed: int,
    channels: int,
    height: int,
    width: int,
    num_classes: int,
):
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(72, channels, height, width, generator=g)
    y = torch.randint(0, num_classes, (72,), generator=g)
    x_tr, y_tr = x[:48], y[:48]
    x_va, y_va = x[48:60], y[48:60]
    x_te, y_te = x[60:], y[60:]
    return datasets_base.DataBundle(
        train_loader=DataLoader(TensorDataset(x_tr, y_tr), batch_size=min(batch_size, len(x_tr)), shuffle=False),
        val_loader=DataLoader(TensorDataset(x_va, y_va), batch_size=min(batch_size, len(x_va)), shuffle=False),
        test_loader=DataLoader(TensorDataset(x_te, y_te), batch_size=min(batch_size, len(x_te)), shuffle=False),
        num_classes=num_classes,
        in_dim=channels * height * width,
        input_shape=(channels, height, width),
        x_test=x_te,
        y_test=y_te,
    )


def _glue_bundle(*, batch_size: int, seed: int, task_name: str):
    g = torch.Generator().manual_seed(int(seed))
    is_regression = str(task_name).lower() == "stsb"
    seq_len = 6

    def _batch(size: int):
        labels = (
            torch.rand(size, generator=g) * 5.0
            if is_regression
            else torch.randint(0, 2, (size,), generator=g)
        )
        return {
            "input_ids": torch.randint(0, 31, (size, seq_len), generator=g, dtype=torch.long),
            "attention_mask": torch.ones(size, seq_len, dtype=torch.long),
            "labels": labels.float() if is_regression else labels.long(),
        }

    return datasets_base.DataBundle(
        train_loader=[_batch(min(batch_size, 8)), _batch(min(batch_size, 8))],
        val_loader=None,
        test_loader=None,
        num_classes=1 if is_regression else 2,
        meta={
            "val_loaders": {"validation": [_batch(min(batch_size, 8))]},
            "num_labels": 1 if is_regression else 2,
            "is_regression": is_regression,
        },
    )


def _patch_supervised_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get_dataset(*, name: str, batch_size: int = 32, seed: int = 0, glue_task: str = "sst2", **_: object):
        if name == "iris":
            return _tabular_bundle(batch_size=batch_size, seed=seed)
        if name == "breast_cancer":
            return _tabular_bundle(batch_size=batch_size, seed=seed, in_dim=30, num_classes=2)
        if name == "mnist":
            return _image_bundle(batch_size=batch_size, seed=seed, channels=1, height=28, width=28, num_classes=10)
        if name == "cifar10":
            return _image_bundle(batch_size=batch_size, seed=seed, channels=3, height=32, width=32, num_classes=10)
        if name == "glue":
            return _glue_bundle(batch_size=batch_size, seed=seed, task_name=glue_task)
        raise AssertionError(f"Unexpected dataset in golden-path test: {name}")

    monkeypatch.setattr(runner_api, "get_dataset", _fake_get_dataset)


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEncoder(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.layer = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])

    class _FakeBase(nn.Module):
        def __init__(self, *, vocab_size: int = 32, hidden_dim: int = 12):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, hidden_dim)
            self.encoder = _FakeEncoder(hidden_dim)

    class _FakeHFClassifier(nn.Module):
        base_model_prefix = "bert"

        def __init__(self, num_labels: int):
            super().__init__()
            self.bert = _FakeBase()
            self.classifier = nn.Linear(12, int(num_labels))
            self.config = SimpleNamespace(num_labels=int(num_labels), vocab_size=32)

        def forward(
            self,
            *,
            input_ids=None,
            attention_mask=None,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            **_kwargs,
        ):
            _ = attention_mask
            if input_ids is None:
                raise ValueError("input_ids is required.")
            h = self.bert.embeddings(input_ids)
            hidden_states = [h]
            for layer in self.bert.encoder.layer:
                h = torch.tanh(layer(h))
                hidden_states.append(h)
            logits = self.classifier(h[:, 0, :])
            out = SimpleNamespace(
                logits=logits,
                hidden_states=tuple(hidden_states) if bool(output_hidden_states) else None,
                last_hidden_state=h,
                loss=None,
            )
            if bool(return_dict):
                return out
            return (logits,)

    class _AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, model_name: str, num_labels: int, trust_remote_code: bool = False):
            _ = (cls, model_name, trust_remote_code)
            return _FakeHFClassifier(num_labels=num_labels)

    tf_mod = types.ModuleType("transformers")
    setattr(tf_mod, "AutoModelForSequenceClassification", _AutoModelForSequenceClassification)
    monkeypatch.setitem(sys.modules, "transformers", tf_mod)


def _run_supervised_config(tmp_path: Path, config_path: Path, *extra_args: str) -> dict[str, object]:
    run_dir = tmp_path / f"{config_path.stem}_run"
    argv = [
        "supervised",
        "--config",
        str(config_path),
        "--epochs",
        "1",
        "--display",
        "none",
        "--run-dir",
        str(run_dir),
        *extra_args,
    ]
    return train_api.run_train_from_argv(argv)


def test_bp_tabular_golden_path_uses_official_quickstart_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_supervised_datasets(monkeypatch)

    summary = _run_supervised_config(tmp_path, REPO_ROOT / "configs" / "train" / "supervised.quickstart.toml")

    assert summary["args"]["dataset"] == "iris"
    assert summary["args"]["model"] == "mlp"
    assert summary["args"]["rule"] == "bp"
    assert summary["train"]["final_epoch"]["train"]["num_batches"] > 0
    assert "test" in summary["eval"]


def test_capsule_optimizer_golden_path_creates_lists_and_trains(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_supervised_datasets(monkeypatch)
    capsule_root = create_api.create_capsule_scaffold(
        capsule_name="golden_capsule",
        base_dir=tmp_path,
        register=False,
    )
    (capsule_root / "optimizers" / "example.py").write_text(
        "import torch\n"
        "from lelabo.optimizers import OptimizerContext, register_optimizer\n\n"
        "@register_optimizer('capsule_sgd')\n"
        "def build_capsule_sgd(ctx: OptimizerContext):\n"
        "    return torch.optim.SGD(ctx.params, lr=ctx.lr, weight_decay=ctx.weight_decay)\n",
        encoding="utf-8",
    )

    plugins.reset_capsule_plugin_cache()
    monkeypatch.chdir(capsule_root)
    try:
        rc = list_cli.main(["optimizers", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "capsule_sgd" in payload["optimizers"]["capsule"]

        summary = _run_supervised_config(
            tmp_path,
            capsule_root / "configs" / "train.supervised.capsule_optimizer.toml",
        )
    finally:
        plugins.reset_capsule_plugin_cache()

    assert summary["args"]["dataset"] == "mnist"
    assert summary["args"]["model"] == "cnn"
    assert summary["args"]["optimizer"] == "capsule_sgd"
    assert summary["args"]["rule"] == "bp"
    assert "test" in summary["eval"]


def test_bp_vision_golden_path_uses_official_cifar10_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_supervised_datasets(monkeypatch)

    summary = _run_supervised_config(tmp_path, REPO_ROOT / "configs" / "train" / "supervised.detailed.toml")

    assert summary["args"]["dataset"] == "cifar10"
    assert summary["args"]["model"] == "cnn"
    assert summary["args"]["rule"] == "bp"
    assert "test" in summary["eval"]


def test_bp_hf_classification_golden_path_uses_official_glue_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_supervised_datasets(monkeypatch)
    _install_fake_transformers(monkeypatch)

    summary = _run_supervised_config(tmp_path, REPO_ROOT / "configs" / "train" / "supervised.glue.toml")

    assert summary["args"]["dataset"] == "glue"
    assert summary["args"]["model"] == "bert"
    assert summary["args"]["rule"] == "bp"
    assert "validation" in summary["eval"]
    assert "acc" in summary["eval"]["validation"]["scalars"]


def test_bp_hf_regression_validation_path_supports_stsb(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_supervised_datasets(monkeypatch)
    _install_fake_transformers(monkeypatch)

    summary = _run_supervised_config(
        tmp_path,
        REPO_ROOT / "configs" / "train" / "supervised.glue.toml",
        "--set",
        "hf.glue_task=stsb",
        "--loss",
        "mse",
        "--metrics",
        "mse,mae",
    )

    assert summary["args"]["glue_task"] == "stsb"
    assert summary["args"]["loss"] == "mse"
    assert "validation" in summary["eval"]
    assert "mse" in summary["eval"]["validation"]["scalars"]
    assert "mae" in summary["eval"]["validation"]["scalars"]


def test_local_rule_golden_path_uses_official_dfa_mnist_mlp_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_supervised_datasets(monkeypatch)

    summary = _run_supervised_config(tmp_path, REPO_ROOT / "configs" / "train" / "supervised.local_rule.toml")

    assert summary["args"]["dataset"] == "mnist"
    assert summary["args"]["model"] == "mlp"
    assert summary["args"]["rule"] == "dfa"
    assert "test" in summary["eval"]


@pytest.mark.parametrize(
    ("rule", "model"),
    [
        ("softhebb", "cnn"),
        ("fa", "mlp"),
        ("drtp", "mlp"),
        ("scl", "mlp"),
    ],
)
def test_secondary_local_rule_paths_have_end_to_end_smoke_coverage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    rule: str,
    model: str,
) -> None:
    _patch_supervised_datasets(monkeypatch)

    summary = _run_supervised_config(
        tmp_path,
        REPO_ROOT / "configs" / "train" / "supervised.local_rule.toml",
        "--rule",
        rule,
        "--model",
        model,
    )

    assert summary["args"]["rule"] == rule
    assert summary["args"]["model"] == model
    assert summary["train"]["final_epoch"]["train"]["num_batches"] > 0
