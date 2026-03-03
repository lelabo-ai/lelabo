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
    conv_mod = importlib.import_module("lelabo.models.builtins.convnet")
    task_mod = importlib.import_module("lelabo.core.task")
    state_mod = importlib.import_module("lelabo.core.state")
    softhebb_mod = importlib.import_module("lelabo.update_rules.builtins.softhebb")
    return conv_mod, task_mod, state_mod, softhebb_mod


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


def test_softhebb_unsup_updates_hidden_then_sup_updates_head(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    conv_mod, task_mod, state_mod, softhebb_mod = _load_modules(monkeypatch)

    model = conv_mod.ConvNetClassifier(
        in_channels=1,
        num_classes=10,
        channels=[8],
        kernel_sizes=3,
        use_bn=False,
        pool_every=1,
    )
    task = task_mod.ClassificationTask(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rule = softhebb_mod.SoftHebb(
        optimizer=optimizer,
        head_optimizer=optimizer,
        unsup_epochs=1,
        conv_t_invert=12.0,
    )

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    state = state_mod.TrainState(epoch=1)
    before_unsup = _snapshot_params(model)
    unsup_stats = rule.train_step(model, task, (x, y), device="cpu", state=state)
    after_unsup = {name: p.detach() for name, p in model.named_parameters()}
    changed_unsup = _changed(before_unsup, after_unsup)

    assert isinstance(unsup_stats, dict)
    assert float(unsup_stats.get("loss", 0.0)) == 0.0
    assert any(name.startswith("conv1.") for name in changed_unsup)
    assert not any(name.startswith("head.") for name in changed_unsup)

    state.epoch = 2
    before_sup = {k: v.clone() for k, v in after_unsup.items()}
    sup_stats = rule.train_step(model, task, (x, y), device="cpu", state=state)
    after_sup = {name: p.detach() for name, p in model.named_parameters()}
    changed_sup = _changed(before_sup, after_sup)

    assert isinstance(sup_stats, dict)
    assert "loss" in sup_stats
    assert any(name.startswith("head.") for name in changed_sup)
    assert not any(name.startswith("conv1.") for name in changed_sup)
