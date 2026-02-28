from __future__ import annotations

import importlib
import sys

import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
initializers_api = importlib.import_module("lelabo.initializers")


def _zero_model_weights(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()


def _clone_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_builtin_initializer_names_include_expected_defaults() -> None:
    names = set(initializers_api.get_initializer_names())
    expected = {
        "none",
        "kaiming_uniform",
        "kaiming_normal",
        "xavier_uniform",
        "xavier_normal",
        "orthogonal",
    }
    assert expected.issubset(names)


def test_initializer_none_is_noop() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    before = _clone_params(model)
    init_fn = initializers_api.make_initializer("none")
    init_fn(model)
    after = _clone_params(model)
    for name, old in before.items():
        assert torch.allclose(old, after[name])


def test_initializer_without_seed_uses_global_rng_state() -> None:
    model_a = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    model_b = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    _zero_model_weights(model_a)
    _zero_model_weights(model_b)

    init_fn = initializers_api.make_initializer("xavier_uniform")

    torch.manual_seed(123)
    init_fn(model_a)
    torch.manual_seed(123)
    init_fn(model_b)

    pa = _clone_params(model_a)
    pb = _clone_params(model_b)
    for name in pa.keys():
        assert torch.allclose(pa[name], pb[name])


def test_initializer_seed_param_overrides_global_rng() -> None:
    model_a = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    model_b = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    _zero_model_weights(model_a)
    _zero_model_weights(model_b)

    init_fn = initializers_api.make_initializer("xavier_uniform", params={"seed": 7})

    torch.manual_seed(123)
    init_fn(model_a)
    torch.manual_seed(999)
    init_fn(model_b)

    pa = _clone_params(model_a)
    pb = _clone_params(model_b)
    for name in pa.keys():
        assert torch.allclose(pa[name], pb[name])
