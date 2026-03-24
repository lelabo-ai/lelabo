from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
wandb_cli = importlib.import_module("lelabo.cli.commands.wandb")


def test_wandb_init_writes_local_project_config(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_wandb_login_state", lambda: (True, "env"))
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: True)

    rc = wandb_cli.main(["init", "--project", "paper-project", "--entity", "research-team", "--enable"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Success: W&B project config initialized." in out
    cfg_path = workspace / ".lelabo" / "config.toml"
    content = cfg_path.read_text(encoding="utf-8")
    assert "[wandb]" in content
    assert 'project = "paper-project"' in content
    assert 'entity = "research-team"' in content
    assert "enabled = true" in content


def test_wandb_init_requires_project_non_interactively(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_wandb_missing"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_is_interactive_tty", lambda: False)

    with pytest.raises(SystemExit) as exc:
        wandb_cli.main(["init"])

    assert "Missing `--project`." in str(exc.value)


def test_wandb_status_reports_missing_project_config(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb_status"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_wandb_package_available", lambda: False)
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: False)
    monkeypatch.setattr(wandb_cli, "_wandb_login_state", lambda: (False, "-"))

    rc = wandb_cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "W&B status" in out
    assert "configured: False" in out
    assert "Run `lelabo wandb init`" in out


def test_wandb_login_fails_cleanly_when_cli_missing(monkeypatch) -> None:
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        wandb_cli.main(["login"])
    assert "W&B CLI is not installed" in str(exc.value)
