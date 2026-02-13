from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from conftest import REPO_ROOT


class _Args:
    lr = 1e-3
    weight_decay = 0.0
    max_grad_norm = 0.5
    model = "mlp"

    def __getattr__(self, _name: str):
        return None


def _stub_namespace_package(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> None:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, pkg)

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, pkg)


def _load_update_rule_modules(monkeypatch: pytest.MonkeyPatch):
    src_root = REPO_ROOT / "src"
    monkeypatch.syspath_prepend(str(src_root))

    # Avoid side effects from __init__.py files that pull optional dependencies.
    _stub_namespace_package(monkeypatch, "lab", src_root / "lab")
    _stub_namespace_package(monkeypatch, "lab.core", src_root / "lab" / "core")
    _stub_namespace_package(monkeypatch, "lab.models", src_root / "lab" / "models")
    _stub_namespace_package(monkeypatch, "lab.algorithms", src_root / "lab" / "algorithms")
    _stub_namespace_package(
        monkeypatch,
        "lab.algorithms.update_rules",
        src_root / "lab" / "algorithms" / "update_rules",
    )

    # Import torch first to avoid environment-specific OpenMP import-order issues.
    importlib.import_module("torch")

    registry = importlib.import_module("lab.algorithms.update_rules.registry")
    importlib.import_module("lab.algorithms.update_rules.builders")
    return registry


def test_all_registered_update_rules_build(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    registry = _load_update_rule_modules(monkeypatch)

    names = registry.get_update_rule_names()
    assert names, "No update rule registered in UPDATE_RULE_REGISTRY."

    param = torch.nn.Parameter(torch.zeros(()), requires_grad=True)
    optimizer = torch.optim.SGD([param], lr=1e-3)
    ctx = registry.UpdateRuleContext(
        args=_Args(),
        model=torch.nn.Identity(),
        task=None,
        optimizer=optimizer,
        mode="supervised",
        dataset="mnist",
    )

    built = {}
    for name in names:
        learner = registry.build_update_rule(name, ctx)
        assert learner is not None, f"Builder returned None for update rule '{name}'."
        built[name] = type(learner).__name__

    assert set(built) == set(names)
