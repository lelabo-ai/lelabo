from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
optimizers_api = importlib.import_module("lelabo.optimizers")
train_api = importlib.import_module("lelabo.cli.commands.train")


def _make_model() -> torch.nn.Module:
    return torch.nn.Linear(4, 2)


def _optimizer_step_updates_params(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> bool:
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    for name, p in model.named_parameters():
        if not torch.allclose(before[name], p.detach()):
            return True
    return False


@pytest.mark.parametrize(
    ("optimizer_name", "expected_type", "kwargs"),
    [
        ("adamw", torch.optim.AdamW, {}),
        ("adam", torch.optim.Adam, {}),
        ("sgd", torch.optim.SGD, {}),
        ("sgd+momentum", torch.optim.SGD, {"momentum": 0.8}),
        ("rmsprop", torch.optim.RMSprop, {}),
        ("adagrad", torch.optim.Adagrad, {}),
    ],
)
def test_builtin_optimizers_are_buildable(
    optimizer_name: str,
    expected_type: type[torch.optim.Optimizer],
    kwargs: dict[str, float],
) -> None:
    model = _make_model()
    optimizer = optimizers_api.make_optimizer(
        optimizer_name,
        model.parameters(),
        lr=0.01,
        weight_decay=0.02,
        **kwargs,
    )
    assert isinstance(optimizer, expected_type)
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(0.01, rel=1e-8)
    assert float(optimizer.param_groups[0]["weight_decay"]) == pytest.approx(0.02, rel=1e-8)
    if optimizer_name == "sgd+momentum":
        assert float(optimizer.param_groups[0]["momentum"]) == pytest.approx(0.8, rel=1e-8)


def test_builtin_optimizer_names_include_core_choices() -> None:
    names = set(optimizers_api.get_optimizer_names())
    assert {"adamw", "sgd", "sgd+momentum", "ano"}.issubset(names)


def test_adamw_forwards_extra_kwargs() -> None:
    model = _make_model()
    optimizer = optimizers_api.make_optimizer(
        "adamw",
        model.parameters(),
        lr=0.01,
        weight_decay=0.02,
        betas=(0.7, 0.97),
        eps=1e-7,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert tuple(optimizer.param_groups[0]["betas"]) == (0.7, 0.97)
    assert float(optimizer.param_groups[0]["eps"]) == pytest.approx(1e-7, rel=1e-8)


@pytest.mark.parametrize(
    ("optimizer_name", "kwargs"),
    [
        ("adamw", {}),
        ("adam", {}),
        ("sgd", {}),
        ("sgd+momentum", {"momentum": 0.8}),
        ("rmsprop", {}),
        ("adagrad", {}),
    ],
)
def test_builtin_optimizers_update_parameters(optimizer_name: str, kwargs: dict[str, float]) -> None:
    model = _make_model()
    optimizer = optimizers_api.make_optimizer(
        optimizer_name,
        model.parameters(),
        lr=0.01,
        weight_decay=0.0,
        **kwargs,
    )
    assert _optimizer_step_updates_params(optimizer, model)


def test_make_optimizer_rejects_empty_name() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="cannot be empty"):
        optimizers_api.make_optimizer(
            "   ",
            model.parameters(),
            lr=0.01,
            weight_decay=0.0,
        )


def test_make_optimizer_rejects_unknown_name() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match=r"\[optimizers\] Unknown"):
        optimizers_api.make_optimizer(
            "does_not_exist",
            model.parameters(),
            lr=0.01,
            weight_decay=0.0,
        )


def test_ano_optimizer_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeAno(torch.optim.Adam):
        pass

    fake_ano_module = types.ModuleType("ano_optimizer")
    setattr(fake_ano_module, "Ano", _FakeAno)
    monkeypatch.setitem(sys.modules, "ano_optimizer", fake_ano_module)

    model = _make_model()
    optimizer = optimizers_api.make_optimizer(
        "ano",
        model.parameters(),
        lr=0.01,
        weight_decay=0.0,
    )
    assert isinstance(optimizer, _FakeAno)
    assert _optimizer_step_updates_params(optimizer, model)


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


def test_optimizer_builder_must_return_torch_optimizer(tmp_path, monkeypatch) -> None:
    capsule_root = tmp_path / "capsule_bad_optimizer"
    (capsule_root / "optimizers").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_bad_optimizer\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "optimizers" / "bad.py").write_text(
        "from lelabo.optimizers import OptimizerContext, register_optimizer\n\n"
        "@register_optimizer('capsule_bad_opt')\n"
        "def build_capsule_bad_opt(ctx: OptimizerContext):\n"
        "    return object()\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(capsule_root)
    model = _make_model()
    with pytest.raises(TypeError, match="must return torch.optim.Optimizer"):
        optimizers_api.make_optimizer(
            "capsule_bad_opt",
            model.parameters(),
            lr=0.01,
            weight_decay=0.0,
        )


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
