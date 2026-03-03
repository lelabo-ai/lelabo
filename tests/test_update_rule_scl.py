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
