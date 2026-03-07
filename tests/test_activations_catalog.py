from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lelabo.activations import (
    CACHE_AUTO_PAIR_ACTIVATION_MODULE_TYPES,
    LOCAL_RULE_SUPPORTED_ACTIVATIONS,
    local_rule_activation_derivative_from_preact,
    local_rule_activation_from_preact,
    local_rule_activation_name_from_value,
    normalize_local_rule_activation_name,
)


def test_local_rule_supported_activations_contract() -> None:
    expected = ("relu", "relu6", "tanh", "sigmoid", "gelu", "silu", "mish", "softsign", "hardsigmoid", "selu", "identity")
    assert LOCAL_RULE_SUPPORTED_ACTIVATIONS == expected


def test_local_rule_activation_name_resolution() -> None:
    assert local_rule_activation_name_from_value(nn.ReLU()) == "relu"
    assert local_rule_activation_name_from_value(nn.GELU()) == "gelu"
    assert local_rule_activation_name_from_value(nn.SiLU()) == "silu"
    assert local_rule_activation_name_from_value(nn.Mish()) == "mish"
    assert local_rule_activation_name_from_value(torch.tanh) == "tanh"
    assert local_rule_activation_name_from_value("SIGMOID") == "sigmoid"
    assert local_rule_activation_name_from_value("gelu") == "gelu"
    assert local_rule_activation_name_from_value("not-supported") is None


def test_local_rule_activation_normalization_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        normalize_local_rule_activation_name("tri")


def test_local_rule_activation_ops_and_derivatives_are_tensor_shaped() -> None:
    preact = torch.linspace(-3.0, 3.0, steps=11)
    for name in LOCAL_RULE_SUPPORTED_ACTIVATIONS:
        act = local_rule_activation_from_preact(name, preact)
        deriv = local_rule_activation_derivative_from_preact(name, preact)
        assert act.shape == preact.shape
        assert deriv.shape == preact.shape
        assert torch.isfinite(act).all()
        assert torch.isfinite(deriv).all()


def test_cache_auto_pair_activation_catalog_contains_common_modules() -> None:
    assert nn.ReLU in CACHE_AUTO_PAIR_ACTIVATION_MODULE_TYPES
    assert nn.Tanh in CACHE_AUTO_PAIR_ACTIVATION_MODULE_TYPES
    assert nn.Sigmoid in CACHE_AUTO_PAIR_ACTIVATION_MODULE_TYPES
