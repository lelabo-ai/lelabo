from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
rl_cfg = importlib.import_module("lab.api.train_rl_config")


def test_parse_rl_param_overrides_normalizes_keys() -> None:
    overrides = rl_cfg.parse_rl_param_overrides(
        ["num-steps=256", "target-kl=0.02", "norm_adv=false"]
    )
    assert overrides == {
        "num_steps": "256",
        "target_kl": "0.02",
        "norm_adv": "false",
    }


def test_parse_rl_param_overrides_rejects_invalid_entry() -> None:
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        rl_cfg.parse_rl_param_overrides(["gamma"])


def test_get_rl_algo_contract_lists_expected_keys() -> None:
    dqn_contract = set(rl_cfg.get_rl_algo_contract("dqn"))
    ppo_contract = set(rl_cfg.get_rl_algo_contract("ppo"))
    assert "batch_size" in dqn_contract
    assert "rl_batch_size" in dqn_contract
    assert "num_steps" in ppo_contract
    assert "ppo_num_steps" in ppo_contract


def test_get_rl_algo_contract_rejects_unknown_algo() -> None:
    with pytest.raises(ValueError, match="Unknown rl algo"):
        rl_cfg.get_rl_algo_contract("missing_algo")


def test_build_dqn_config_overrides_defaults() -> None:
    cfg = rl_cfg.build_dqn_config(
        {
            "gamma": "0.97",
            "batch_size": "64",
            "eps_end": "0.01",
        }
    )
    assert cfg.gamma == pytest.approx(0.97)
    assert cfg.batch_size == 64
    assert cfg.eps_end == pytest.approx(0.01)


def test_build_dqn_config_accepts_legacy_alias() -> None:
    cfg = rl_cfg.build_dqn_config({"rl_batch_size": "96"})
    assert cfg.batch_size == 96


def test_build_dqn_config_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported DQN override"):
        rl_cfg.build_dqn_config({"num_steps": "128"})


def test_build_ppo_config_overrides_top_and_nested_defaults() -> None:
    cfg = rl_cfg.build_ppo_config(
        {
            "num_steps": "256",
            "gamma": "0.98",
            "clip_coef": "0.1",
            "norm_adv": "false",
            "target_kl": "0.02",
        }
    )
    assert cfg.num_steps == 256
    assert cfg.gamma == pytest.approx(0.98)
    assert cfg.ppo.clip_coef == pytest.approx(0.1)
    assert cfg.ppo.norm_adv is False
    assert cfg.ppo.target_kl == pytest.approx(0.02)


def test_build_ppo_config_accepts_legacy_alias() -> None:
    cfg = rl_cfg.build_ppo_config({"ppo_num_minibatches": "8"})
    assert cfg.num_minibatches == 8


def test_build_ppo_config_accepts_explicit_none_for_optional_float() -> None:
    cfg = rl_cfg.build_ppo_config({"target_kl": "none"})
    assert cfg.ppo.target_kl is None


def test_build_ppo_config_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unsupported PPO override"):
        rl_cfg.build_ppo_config({"eps_decay_steps": "1000"})
