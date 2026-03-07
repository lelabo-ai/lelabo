from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

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
    task_mod = importlib.import_module("lelabo.core.task")
    dni_mod = importlib.import_module("lelabo.update_rules.builtins.dni")
    return mlp_mod, task_mod, dni_mod


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


def test_dni_updates_mlp_hidden_and_head(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    mlp_mod, task_mod, dni_mod = _load_modules(monkeypatch)

    model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")
    task = task_mod.ClassificationTask(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rule = dni_mod.DNI(
        optimizer=optimizer,
        sg_lr=1e-3,
        sg_hidden=0,
        condition_on_label=True,
        lambda_mix=1.0,
        sg_scale=1.0,
        activation="relu",
    )

    x = torch.randn(6, 8)
    y = torch.randint(0, 3, (6,))

    before = _snapshot_params(model)
    stats_1 = rule.train_step(model, task, (x, y), device="cpu")
    after = {name: p.detach() for name, p in model.named_parameters()}
    changed = _changed(before, after)

    assert isinstance(stats_1, dict)
    assert "loss" in stats_1
    assert any(name.startswith("net.layer0.") for name in changed)
    assert any(name.startswith("net.head.") for name in changed)


def test_dni_invalid_activation_raises_on_train_step(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    mlp_mod, task_mod, dni_mod = _load_modules(monkeypatch)

    model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")
    task = task_mod.ClassificationTask(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rule = dni_mod.DNI(
        optimizer=optimizer,
        activation="not-a-real-activation",
    )

    x = torch.randn(6, 8)
    y = torch.randint(0, 3, (6,))
    with pytest.raises(ValueError, match="Unsupported activation"):
        rule.train_step(model, task, (x, y), device="cpu")


def test_dni_state_dict_roundtrip_preserves_sg_state(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    mlp_mod, task_mod, dni_mod = _load_modules(monkeypatch)

    model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")
    task = task_mod.ClassificationTask(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rule = dni_mod.DNI(
        optimizer=optimizer,
        sg_lr=1e-3,
        sg_hidden=8,
        condition_on_label=True,
        lambda_mix=0.5,
        sg_scale=1.0,
        activation="relu",
    )

    x = torch.randn(6, 8)
    y = torch.randint(0, 3, (6,))
    _ = rule.train_step(model, task, (x, y), device="cpu")
    state = rule.state_dict()

    optimizer_2 = torch.optim.SGD(model.parameters(), lr=1e-3)
    reloaded = dni_mod.DNI(
        optimizer=optimizer_2,
        sg_lr=1e-3,
        sg_hidden=8,
        condition_on_label=True,
        lambda_mix=0.5,
        sg_scale=1.0,
        activation="relu",
    )
    reloaded.load_state_dict(state)

    assert reloaded.global_step == rule.global_step
    assert reloaded._sg_signatures == rule._sg_signatures
    assert set(reloaded._sg_models.keys()) == set(rule._sg_models.keys())
