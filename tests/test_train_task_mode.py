from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
train_api = importlib.import_module("lelabo.api.train")
update_rules_api = importlib.import_module("lelabo.update_rules")
lab_pkg = importlib.import_module("lelabo")


def test_supervised_mode_normalization_sets_task() -> None:
    args = Namespace(mode="supervised", dataset="iris")
    out = train_api._normalize_train_args(args)
    assert out.task == "supervised"
    assert out.dataset == "iris"


def test_rl_mode_normalization_sets_task_and_dataset() -> None:
    args = Namespace(mode="rl", env="CartPole-v1")
    out = train_api._normalize_train_args(args)
    assert out.task == "rl"
    assert out.env == "CartPole-v1"
    assert out.dataset == "env:CartPole-v1"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown train mode"):
        train_api._normalize_train_args(Namespace(mode="auto", source="iris"))


def test_legacy_unified_source_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown train mode"):
        train_api._normalize_train_args(Namespace(source="iris"))


def test_supervised_default_algo_is_registered() -> None:
    args = train_api.parse_train_args(["supervised", "--dataset", "iris"])
    available = set(update_rules_api.get_update_rule_names())
    assert args.algo in available
    assert args.config_version == "1.0"
    assert isinstance(args.lelabo_version, str)
    assert args.lelabo_version.strip()


def test_supervised_loss_override_is_exposed() -> None:
    args = train_api.parse_train_args(["supervised", "--dataset", "iris", "--loss", "bce"])
    assert args.loss == "bce"


def test_supervised_display_override_is_exposed() -> None:
    args = train_api.parse_train_args(["supervised", "--dataset", "iris", "--display", "none"])
    assert args.display == "none"


def test_supervised_initializer_override_is_exposed() -> None:
    args = train_api.parse_train_args(["supervised", "--dataset", "iris", "--initializer", "xavier_uniform"])
    assert args.initializer == "xavier_uniform"


def test_supervised_set_override_parses_nested_field() -> None:
    args = train_api.parse_train_args(
        ["supervised", "--dataset", "iris", "--set", "robustness.max_samples=123"]
    )
    assert args.robustness_max_samples == 123


def test_supervised_config_file_is_loaded(tmp_path: Path) -> None:
    cfg = tmp_path / "train.toml"
    cfg.write_text(
        "\n".join(
            [
                "task = \"supervised\"",
                "",
                "[dataset]",
                "name = \"iris\"",
                "",
                "[optimizer]",
                "name = \"adamw\"",
                "[optimizer.params]",
                "lr = 0.004",
            ]
        ),
        encoding="utf-8",
    )
    args = train_api.parse_train_args(["supervised", "--config", str(cfg)])
    assert args.dataset == "iris"
    assert args.lr == pytest.approx(0.004)


def test_runtime_display_config_is_exposed(tmp_path: Path) -> None:
    cfg = tmp_path / "train.toml"
    cfg.write_text(
        "\n".join(
            [
                'task = "supervised"',
                "",
                "[dataset]",
                'name = "iris"',
                "",
                "[runtime]",
                'display = "none"',
            ]
        ),
        encoding="utf-8",
    )
    args = train_api.parse_train_args(["supervised", "--config", str(cfg)])
    assert args.display == "none"


def test_runtime_verbose_is_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "train.toml"
    cfg.write_text(
        "\n".join(
            [
                'task = "supervised"',
                "",
                "[dataset]",
                'name = "iris"',
                "",
                "[runtime]",
                "verbose = 0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime.verbose has been removed"):
        train_api.parse_train_args(["supervised", "--config", str(cfg)])


def test_supervised_set_override_wins(tmp_path: Path) -> None:
    cfg = tmp_path / "train.toml"
    cfg.write_text(
        "\n".join(
            [
                "task = \"supervised\"",
                "",
                "[dataset]",
                "name = \"iris\"",
                "",
                "[model]",
                "name = \"mlp\"",
                "[model.params]",
                "hidden = 256",
            ]
        ),
        encoding="utf-8",
    )
    args = train_api.parse_train_args(
        [
            "supervised",
            "--config",
            str(cfg),
            "--set",
            "model.params.hidden=1024",
        ]
    )
    assert args.hidden == 1024


def test_supervised_auto_discovers_project_train_toml(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "train.toml"
    cfg.write_text(
        "\n".join(
            [
                "task = \"supervised\"",
                "",
                "[dataset]",
                "name = \"iris\"",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    args = train_api.parse_train_args(["supervised"])
    assert args.dataset == "iris"


def test_rejects_unsupported_config_version(tmp_path: Path) -> None:
    cfg = tmp_path / "train.toml"
    cfg.write_text(
        "\n".join(
            [
                "config_version = \"9.9\"",
                "task = \"supervised\"",
                "",
                "[dataset]",
                "name = \"iris\"",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported config_version"):
        train_api.parse_train_args(["supervised", "--config", str(cfg)])


def test_auto_version_tokens_resolve_to_runtime_values(tmp_path: Path) -> None:
    cfg = tmp_path / "train.toml"
    cfg.write_text(
        "\n".join(
            [
                "config_version = \"auto\"",
                "lelabo_version = \"auto\"",
                "task = \"supervised\"",
                "",
                "[dataset]",
                "name = \"iris\"",
            ]
        ),
        encoding="utf-8",
    )
    args = train_api.parse_train_args(["supervised", "--config", str(cfg)])
    assert args.config_version == "1.0"
    assert args.lelabo_version == lab_pkg.__version__
