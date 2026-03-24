"""Tests for the W&B integration layer in RunLogger and config resolution."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lelabo.config.schema import WandbSpec
from lelabo.core.logger import RunLogger


def test_wandb_spec_defaults():
    spec = WandbSpec()
    assert spec.project is None
    assert spec.entity is None
    assert spec.tags == ()
    assert spec.group is None
    assert spec.notes == ""
    assert spec.enabled is True


def test_wandb_spec_in_supervised_config():
    from lelabo.config.resolve import resolve_supervised_config

    cfg = resolve_supervised_config(
        config_path=None,
        cli_overrides={"dataset": {"name": "iris"}},
        set_overrides=["wandb.project=my-project", "wandb.entity=my-team"],
    )
    assert cfg.wandb.project == "my-project"
    assert cfg.wandb.entity == "my-team"
    assert cfg.wandb.enabled is True


def test_wandb_env_var_fallback():
    with patch.dict(os.environ, {"WANDB_PROJECT": "env-project", "WANDB_ENTITY": "env-team"}):
        from lelabo.config.resolve import resolve_supervised_config

        cfg = resolve_supervised_config(
            config_path=None,
            cli_overrides={"dataset": {"name": "iris"}},
        )
        assert cfg.wandb.project == "env-project"
        assert cfg.wandb.entity == "env-team"


def test_wandb_config_overrides_env():
    with patch.dict(os.environ, {"WANDB_PROJECT": "env-project"}):
        from lelabo.config.resolve import resolve_supervised_config

        cfg = resolve_supervised_config(
            config_path=None,
            cli_overrides={"dataset": {"name": "iris"}},
            set_overrides=["wandb.project=config-project"],
        )
        assert cfg.wandb.project == "config-project"


def test_wandb_project_defaults_override_env():
    with patch.dict(os.environ, {"WANDB_PROJECT": "env-project", "WANDB_ENTITY": "env-team"}):
        from lelabo.config.resolve import resolve_supervised_config

        cfg = resolve_supervised_config(
            config_path=None,
            cli_overrides={"dataset": {"name": "iris"}},
            project_defaults={"wandb": {"project": "local-project", "entity": "local-team"}},
        )
        assert cfg.wandb.project == "local-project"
        assert cfg.wandb.entity == "local-team"


def test_wandb_run_config_overrides_project_defaults(tmp_path: Path):
    config_path = tmp_path / "train.supervised.toml"
    config_path.write_text(
        "\n".join(
            [
                '[dataset]',
                'name = "iris"',
                '',
                '[wandb]',
                'project = "config-project"',
                'entity = "config-team"',
                '',
            ]
        ),
        encoding="utf-8",
    )
    from lelabo.config.resolve import resolve_supervised_config

    cfg = resolve_supervised_config(
        config_path=config_path,
        project_defaults={"wandb": {"project": "local-project", "entity": "local-team"}},
    )
    assert cfg.wandb.project == "config-project"
    assert cfg.wandb.entity == "config-team"


def test_train_parser_inherits_local_project_wandb_config(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace_wandb_train"
    workspace.mkdir()
    local_cfg = workspace / ".lelabo" / "config.toml"
    local_cfg.parent.mkdir(parents=True, exist_ok=True)
    local_cfg.write_text(
        "\n".join(
            [
                "[wandb]",
                'project = "workspace-project"',
                'entity = "workspace-team"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    from lelabo.cli.commands.train import parse_train_args

    args = parse_train_args(["supervised", "--dataset", "iris", "--model", "mlp", "--rule", "bp"])
    assert args.wandb_project == "workspace-project"
    assert args.wandb_entity == "workspace-team"
    assert args.wandb_enabled is True


def test_logger_without_wandb_does_not_error(tmp_path: Path):
    logger = RunLogger(run_dir=tmp_path, wandb_project=None)
    logger.write_meta({"dataset": "iris"}, task="supervised")
    logger.log({"t": "epoch", "epoch": 1, "train.loss": 0.5})
    logger.write_summary({"args": {}}, status="succeeded")
    logger.finalize_meta("succeeded")
    assert (tmp_path / "meta.json").exists()
    assert (tmp_path / "metrics.jsonl").exists()


def test_logger_wandb_project_set_but_no_wandb_installed(tmp_path: Path):
    """Logger should gracefully handle wandb not being importable."""
    import importlib
    import sys

    # Temporarily remove wandb from imports if present
    original = sys.modules.get("wandb")
    sys.modules["wandb"] = None  # type: ignore[assignment]
    try:
        logger = RunLogger(run_dir=tmp_path, wandb_project="test-project")
        assert not logger._wandb_available
        logger.write_meta({"dataset": "iris"}, task="supervised")
        logger.log({"t": "epoch", "epoch": 1, "train.loss": 0.5})
        logger.finalize_meta("succeeded")
    finally:
        if original is not None:
            sys.modules["wandb"] = original
        else:
            sys.modules.pop("wandb", None)


def test_wandb_fields_in_namespace():
    from lelabo.config.resolve import resolve_supervised_config, to_supervised_namespace

    cfg = resolve_supervised_config(
        config_path=None,
        cli_overrides={"dataset": {"name": "iris"}},
        set_overrides=["wandb.project=test-proj", "wandb.group=test-grp"],
    )
    ns = to_supervised_namespace(cfg, device_resolver=lambda: "cpu")
    assert ns.wandb_project == "test-proj"
    assert ns.wandb_group == "test-grp"
    assert ns.wandb_enabled is True
