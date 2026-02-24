from __future__ import annotations

import importlib
import sys

import pytest
import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
optimizers_api = importlib.import_module("lelabo.optimizers")
train_api = importlib.import_module("lelabo.api.train")


def _make_model() -> torch.nn.Module:
    return torch.nn.Linear(4, 2)


def test_builtin_adamw_is_buildable() -> None:
    model = _make_model()
    optimizer = optimizers_api.make_optimizer(
        "adamw",
        model.parameters(),
        lr=0.01,
        weight_decay=0.02,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(0.01, rel=1e-8)
    assert float(optimizer.param_groups[0]["weight_decay"]) == pytest.approx(0.02, rel=1e-8)


def test_builtin_optimizer_names_include_core_choices() -> None:
    names = set(optimizers_api.get_optimizer_names())
    assert {"adamw", "sgd", "sgd+momentum", "ano"}.issubset(names)


def test_builtin_momentum_alias_is_buildable() -> None:
    model = _make_model()
    optimizer = optimizers_api.make_optimizer(
        "sgd+momentum",
        model.parameters(),
        lr=0.05,
        momentum=0.8,
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert float(optimizer.param_groups[0]["momentum"]) == pytest.approx(0.8, rel=1e-8)


def test_optimizer_can_be_loaded_from_capsule_plugin(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_with_optimizer"
    (capsule_root / "optimizers").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_with_optimizer\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "optimizers" / "example.py").write_text(
        "import torch\n"
        "from lelabo.optimizers import OptimizerContext, register_optimizer\n\n"
        "@register_optimizer('capsule_adam')\n"
        "def build_capsule_adam(ctx: OptimizerContext):\n"
        "    return torch.optim.Adam(ctx.params, lr=ctx.lr, weight_decay=ctx.weight_decay)\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(capsule_root)
    model = _make_model()
    optimizer = optimizers_api.make_optimizer(
        "capsule_adam",
        model.parameters(),
        lr=0.02,
        weight_decay=0.03,
    )
    assert isinstance(optimizer, torch.optim.Adam)
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(0.02, rel=1e-8)
    assert float(optimizer.param_groups[0]["weight_decay"]) == pytest.approx(0.03, rel=1e-8)


def test_train_parser_accepts_capsule_optimizer_choice(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_with_optimizer_choice"
    (capsule_root / "optimizers").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_with_optimizer_choice\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "optimizers" / "choice.py").write_text(
        "import torch\n"
        "from lelabo.optimizers import OptimizerContext, register_optimizer\n\n"
        "@register_optimizer('capsule_sgd')\n"
        "def build_capsule_sgd(ctx: OptimizerContext):\n"
        "    return torch.optim.SGD(ctx.params, lr=ctx.lr, weight_decay=ctx.weight_decay)\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(capsule_root)
    args = train_api.parse_train_args(
        ["supervised", "--dataset", "iris", "--optimizer", "capsule_sgd"]
    )
    assert args.optimizer == "capsule_sgd"
