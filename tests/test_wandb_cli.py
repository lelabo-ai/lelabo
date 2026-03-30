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


def test_wandb_init_interactive_suggests_repo_name(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_repo_name"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_wandb_login_state", lambda: (True, "env"))
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: True)
    monkeypatch.setattr(wandb_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(wandb_cli, "_suggest_project_name", lambda: "repo-suggested")
    prompts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        wandb_cli,
        "_prompt_with_default",
        lambda label, default: prompts.append((label, default)) or default,
    )
    monkeypatch.setattr(wandb_cli, "_confirm", lambda question, default=False: True)

    rc = wandb_cli.main(["init"])
    assert rc == 0
    _ = capsys.readouterr()
    assert prompts
    assert prompts[0] == ("W&B project", "repo-suggested")


def test_resolve_entity_prefers_config_over_env_and_login(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "env-team")
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["login-user", "team-a"])
    info = wandb_cli._resolve_entity({"entity": "config-team"})
    assert info["configured_entity"] == "config-team"
    assert info["env_entity"] == "env-team"
    assert info["effective_entity"] == "config-team"
    assert info["source"] == "config"


def test_resolve_entity_uses_env_when_config_missing(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "env-team")
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["login-user"])
    info = wandb_cli._resolve_entity({})
    assert info["effective_entity"] == "env-team"
    assert info["source"] == "env"


def test_resolve_entity_uses_login_when_unique(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["login-user"])
    info = wandb_cli._resolve_entity({})
    assert info["effective_entity"] == "login-user"
    assert info["source"] == "login"


def test_resolve_entity_leaves_ambiguous_login_candidates_unset(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["login-user", "team-a"])
    info = wandb_cli._resolve_entity({})
    assert info["effective_entity"] == ""
    assert info["source"] == "none"


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


def test_wandb_status_shows_dashboard_url(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb_status_url"
    workspace.mkdir()
    local_cfg = workspace / ".lelabo" / "config.toml"
    local_cfg.parent.mkdir(parents=True, exist_ok=True)
    local_cfg.write_text(
        "\n".join(
            [
                "[wandb]",
                'project = "demo-project"',
                'entity = "demo-team"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_wandb_package_available", lambda: True)
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: True)
    monkeypatch.setattr(wandb_cli, "_wandb_login_state", lambda: (True, "env"))
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["demo-team"])

    rc = wandb_cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dashboard_url: https://wandb.ai/demo-team/demo-project" in out
    assert "effective_entity: demo-team" in out


def test_wandb_init_uses_unique_login_entity_non_interactively(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb_auto_entity"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_is_interactive_tty", lambda: False)
    monkeypatch.setattr(wandb_cli, "_wandb_login_state", lambda: (True, "env"))
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["login-user"])

    rc = wandb_cli.main(["init", "--project", "paper-project", "--enable"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Success: W&B project config initialized." in out
    content = (workspace / ".lelabo" / "config.toml").read_text(encoding="utf-8")
    assert 'entity = "login-user"' in content


def test_wandb_init_interactive_uses_picker_when_multiple_entities(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb_picker"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(wandb_cli, "_wandb_login_state", lambda: (True, "env"))
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: True)
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["user-a", "team-z"])
    monkeypatch.setattr(wandb_cli, "_suggest_project_name", lambda: "repo-suggested")
    monkeypatch.setattr(wandb_cli, "_pick_entity_interactive", lambda candidates: "team-z")
    monkeypatch.setattr(
        wandb_cli,
        "_prompt_with_default",
        lambda label, default: default,
    )
    monkeypatch.setattr(wandb_cli, "_confirm", lambda question, default=False: True)

    rc = wandb_cli.main(["init"])
    assert rc == 0
    _ = capsys.readouterr()
    content = (workspace / ".lelabo" / "config.toml").read_text(encoding="utf-8")
    assert 'entity = "team-z"' in content


def test_wandb_login_fails_cleanly_when_cli_missing(monkeypatch) -> None:
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        wandb_cli.main(["login"])
    assert "W&B CLI is not installed" in str(exc.value)


def test_wandb_logout_fails_cleanly_when_cli_missing(monkeypatch) -> None:
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        wandb_cli.main(["logout"])
    assert "W&B CLI is not installed" in str(exc.value)


def test_wandb_logout_keeps_project_config(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb_logout"
    workspace.mkdir()
    local_cfg = workspace / ".lelabo" / "config.toml"
    local_cfg.parent.mkdir(parents=True, exist_ok=True)
    local_cfg.write_text(
        "\n".join(
            [
                "[wandb]",
                'project = "demo-project"',
                'entity = "demo-team"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: True)
    monkeypatch.setattr(
        wandb_cli.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"returncode": 0})(),
    )

    rc = wandb_cli.main(["logout"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Success: W&B logout completed." in out
    assert "Project-local W&B config remains saved" in out


def test_wandb_open_uses_env_entity_when_config_missing_it(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb_open_entity"
    workspace.mkdir()
    local_cfg = workspace / ".lelabo" / "config.toml"
    local_cfg.parent.mkdir(parents=True, exist_ok=True)
    local_cfg.write_text(
        "\n".join(
            [
                "[wandb]",
                'project = "demo-project"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("WANDB_ENTITY", "env-team")
    opened: list[str] = []
    monkeypatch.setattr(wandb_cli.webbrowser, "open", lambda url: opened.append(url) or True)

    rc = wandb_cli.main(["open"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Using W&B entity from env: env-team" in out
    assert opened == ["https://wandb.ai/env-team/demo-project"]


def test_wandb_open_fails_when_login_entities_are_ambiguous(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_wandb_open_ambiguous"
    workspace.mkdir()
    local_cfg = workspace / ".lelabo" / "config.toml"
    local_cfg.parent.mkdir(parents=True, exist_ok=True)
    local_cfg.write_text(
        "\n".join(
            [
                "[wandb]",
                'project = "demo-project"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["user-a", "team-z"])

    with pytest.raises(SystemExit) as exc:
        wandb_cli.main(["open"])
    assert "Could not choose a single W&B entity automatically" in str(exc.value)


def test_wandb_open_prints_url_and_opens_browser(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb_open"
    workspace.mkdir()
    local_cfg = workspace / ".lelabo" / "config.toml"
    local_cfg.parent.mkdir(parents=True, exist_ok=True)
    local_cfg.write_text(
        "\n".join(
            [
                "[wandb]",
                'project = "demo-project"',
                'entity = "demo-team"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_detect_wandb_login_entities", lambda: ["demo-team"])
    opened: list[str] = []
    monkeypatch.setattr(wandb_cli.webbrowser, "open", lambda url: opened.append(url) or True)

    rc = wandb_cli.main(["open"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "https://wandb.ai/demo-team/demo-project" in out
    assert opened == ["https://wandb.ai/demo-team/demo-project"]


def test_wandb_sync_defaults_to_local_wandb_dir(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_wandb_sync"
    workspace.mkdir()
    (workspace / "wandb").mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(wandb_cli, "_wandb_cli_available", lambda: True)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wandb_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(list(cmd)) or type("Proc", (), {"returncode": 0})(),
    )

    rc = wandb_cli.main(["sync"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Success: W&B sync completed." in out
    assert calls == [["wandb", "sync", str((workspace / "wandb").resolve())]]


def test_wandb_sync_fails_when_path_missing(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_wandb_sync_missing"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit) as exc:
        wandb_cli.main(["sync"])
    assert "W&B sync path not found" in str(exc.value)
