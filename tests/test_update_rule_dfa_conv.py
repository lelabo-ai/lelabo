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
    dfa_mod = importlib.import_module("lelabo.update_rules.builtins.dfa")
    return conv_mod, task_mod, dfa_mod


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


def test_dfa_updates_conv_and_head(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    conv_mod, task_mod, dfa_mod = _load_modules(monkeypatch)

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
    rule = dfa_mod.DirectFeedbackAlignment(optimizer=optimizer, activation_name="relu")

    x = torch.randn(6, 1, 28, 28)
    y = torch.randint(0, 10, (6,))

    before = _snapshot_params(model)
    stats = rule.train_step(model, task, (x, y), device="cpu")
    after = {name: p.detach() for name, p in model.named_parameters()}
    changed = _changed(before, after)

    assert isinstance(stats, dict)
    assert any(name.startswith("conv1.") for name in changed)
    assert any(name.startswith("head.") for name in changed)
