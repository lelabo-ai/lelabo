from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from conftest import REPO_ROOT


def _stub_namespace_package(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> None:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, pkg)

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, pkg)


def _load_modules(monkeypatch: pytest.MonkeyPatch):
    src_root = REPO_ROOT / "src"
    monkeypatch.syspath_prepend(str(src_root))

    _stub_namespace_package(monkeypatch, "lelabo", src_root / "lelabo")
    _stub_namespace_package(monkeypatch, "lelabo.core", src_root / "lelabo" / "core")
    _stub_namespace_package(monkeypatch, "lelabo.models", src_root / "lelabo" / "models")
    _stub_namespace_package(monkeypatch, "lelabo.update_rules", src_root / "lelabo" / "update_rules")

    importlib.import_module("torch")
    mlp_mod = importlib.import_module("lelabo.models.builtins.mlp")
    task_mod = importlib.import_module("lelabo.supervised.tasks")
    scl_mod = importlib.import_module("lelabo.update_rules.builtins.scl")
    return mlp_mod, task_mod, scl_mod


def _snapshot_params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _changed(before, after):
    names = set()
    for name, old in before.items():
        new = after.get(name)
        if new is None:
            continue
        if not torch.allclose(old, new, atol=0.0, rtol=0.0):
            names.add(name)
    return names


def test_scl_updates_hidden_and_head(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    mlp_mod, task_mod, scl_mod = _load_modules(monkeypatch)

    model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")
    task = task_mod.ClassificationTask(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rule = scl_mod.SoftContrastiveLearning(
        optimizer=optimizer,
        supcon_tau=0.1,
        local_lr=1e-3,
        local_optim="adamw",
        proj_dim=8,
    )

    x = torch.randn(8, 8)
    y = torch.randint(0, 3, (8,))

    before = _snapshot_params(model)
    stats = rule.train_step(model, task, (x, y), device="cpu")
    after = {name: p.detach() for name, p in model.named_parameters()}
    changed = _changed(before, after)

    assert isinstance(stats, dict)
    assert "loss" in stats
    assert "supcon_loss" in stats
    assert any(name.startswith("net.layer0.") for name in changed)
    assert any(name.startswith("net.head.") for name in changed)


def test_scl_defaults_to_runner_optimizer_for_local_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    mlp_mod, task_mod, scl_mod = _load_modules(monkeypatch)

    model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")
    task = task_mod.ClassificationTask(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-2)
    rule = scl_mod.SoftContrastiveLearning(
        optimizer=optimizer,
        supcon_tau=0.1,
        proj_dim=8,
    )

    x = torch.randn(8, 8)
    y = torch.randint(0, 3, (8,))
    _ = rule.train_step(model, task, (x, y), device="cpu")

    assert rule._local_opt_by_name
    for opt in rule._local_opt_by_name.values():
        assert isinstance(opt, optimizer.__class__)
        assert abs(float(opt.param_groups[0]["lr"]) - 7e-4) < 1e-12


def test_scl_updates_transformer_like_hidden_block(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    _, task_mod, scl_mod = _load_modules(monkeypatch)

    @dataclass
    class _BlockSpec:
        name: str
        module: nn.Module
        rep: str = "cls"
        is_output: bool = False

    class _TinyTransformerLayer(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.ff = nn.Linear(dim, dim)

        def forward(self, x, attention_mask=None):  # noqa: ARG002
            return torch.tanh(self.ff(x))

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = _TinyTransformerLayer(8)
            self.head = nn.Linear(8, 3)

        def get_blocks(self):
            return [
                _BlockSpec(name="encoder.layer0", module=self.layer0, rep="cls", is_output=False),
                _BlockSpec(name="head", module=self.head, rep="cls", is_output=True),
            ]

        def forward(self, x):
            h = self.layer0(x)
            return self.head(h[:, 0, :])

    model = _TinyModel()
    task = task_mod.ClassificationTask(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rule = scl_mod.SoftContrastiveLearning(
        optimizer=optimizer,
        supcon_tau=0.1,
        proj_dim=8,
    )

    x = torch.randn(8, 5, 8)
    y = torch.randint(0, 3, (8,))
    before = _snapshot_params(model)
    stats = rule.train_step(model, task, (x, y), device="cpu")
    after = {name: p.detach() for name, p in model.named_parameters()}
    changed = _changed(before, after)

    assert isinstance(stats, dict)
    assert "loss" in stats
    assert "supcon_loss" in stats
    assert any(name.startswith("layer0.") for name in changed)
    assert any(name.startswith("head.") for name in changed)


def test_scl_mapping_batch_hf_like_uses_hidden_state_local_view(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    _, task_mod, scl_mod = _load_modules(monkeypatch)

    @dataclass
    class _BlockSpec:
        name: str
        module: nn.Module
        rep: str = "cls"
        is_output: bool = False

    class _TinyTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ff = nn.Linear(8, 8)

        def forward(self, x, attention_mask=None):  # noqa: ARG002
            return torch.tanh(self.ff(x))

    class _TinyHFLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(32, 8)
            self.layer0 = _TinyTransformerLayer()
            self.head = nn.Linear(8, 3)
            # Trigger HF mode in SCL.
            self.bert = SimpleNamespace(
                get_extended_attention_mask=lambda attn, input_shape, device=None: attn  # noqa: ARG005
            )

        def get_blocks(self):
            return [
                _BlockSpec(name="embeddings", module=self.embeddings, rep="cls", is_output=False),
                _BlockSpec(name="encoder.layer0", module=self.layer0, rep="cls", is_output=False),
                _BlockSpec(name="head", module=self.head, rep="cls", is_output=True),
            ]

        def forward(self, *, input_ids, attention_mask=None, labels=None, return_cache=False):  # noqa: ARG002
            h0 = self.embeddings(input_ids)
            h1 = self.layer0(h0)
            logits = self.head(h1[:, 0, :])
            out = SimpleNamespace(logits=logits)
            if return_cache:
                cache = {
                    "hidden_states": (h0, h1),
                    "module_inputs": {
                        "embeddings": input_ids,
                        "encoder.layer0": h0,
                        "head": h1[:, 0, :],
                    },
                }
                return out, cache
            return out

    model = _TinyHFLike()
    task = task_mod.ClassificationTask(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rule = scl_mod.SoftContrastiveLearning(
        optimizer=optimizer,
        supcon_tau=0.1,
        proj_dim=8,
    )

    batch = {
        "input_ids": torch.randint(0, 31, (8, 5), dtype=torch.long),
        "attention_mask": torch.ones(8, 5, dtype=torch.long),
        "labels": torch.randint(0, 3, (8,), dtype=torch.long),
    }

    before = _snapshot_params(model)
    stats = rule.train_step(model, task, batch, device="cpu")
    after = {name: p.detach() for name, p in model.named_parameters()}
    changed = _changed(before, after)

    assert isinstance(stats, dict)
    assert "loss" in stats
    assert "supcon_loss" in stats
    assert any(name.startswith("layer0.") for name in changed)
    assert any(name.startswith("head.") for name in changed)


def test_scl_state_dict_roundtrip_preserves_local_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    mlp_mod, task_mod, scl_mod = _load_modules(monkeypatch)

    model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")
    task = task_mod.ClassificationTask(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rule = scl_mod.SoftContrastiveLearning(
        optimizer=optimizer,
        supcon_tau=0.1,
        proj_dim=8,
    )

    x = torch.randn(8, 8)
    y = torch.randint(0, 3, (8,))
    _ = rule.train_step(model, task, (x, y), device="cpu")
    state = rule.state_dict()

    optimizer_2 = torch.optim.SGD(model.parameters(), lr=1e-3)
    reloaded = scl_mod.SoftContrastiveLearning(
        optimizer=optimizer_2,
        supcon_tau=0.1,
        proj_dim=8,
    )
    reloaded.load_state_dict(state)

    assert reloaded.global_step == rule.global_step
    assert set(reloaded._proj_by_name.keys()) == set(rule._proj_by_name.keys())
    assert set(reloaded._pending_local_opt_state_by_name.keys()) == set(rule._local_opt_by_name.keys())
