from __future__ import annotations

import importlib
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
config_cli = importlib.import_module("lelabo.cli.commands.config")


def test_config_cli_set_get_show_with_explicit_file(tmp_path, capsys) -> None:
    cfg = tmp_path / "settings.toml"

    rc = config_cli.main(["set", "github.owner", "acme", "--file", str(cfg)])
    assert rc == 0
    assert cfg.exists()
    capsys.readouterr()

    rc = config_cli.main(["set", "capsules.install_checkout", "true", "--file", str(cfg)])
    assert rc == 0
    capsys.readouterr()

    rc = config_cli.main(["get", "github.owner", "--file", str(cfg)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "acme"

    rc = config_cli.main(["show", "--file", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[github]" in out
    assert 'owner = "acme"' in out
    assert "[capsules]" in out
    assert "install_checkout = true" in out


def test_config_cli_show_effective_prefers_local_override(tmp_path, monkeypatch, capsys) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    global_cfg = xdg / "lelabo" / "config.toml"
    global_cfg.parent.mkdir(parents=True, exist_ok=True)
    global_cfg.write_text(
        '\n'.join(
            [
                "[github]",
                'owner = "global-owner"',
                'default_visibility = "private"',
                'default_branch = "main"',
                "create_repo_if_missing = true",
                "",
                "[capsules]",
                'store_dir = ""',
                'default_checkout_dir = "."',
                "install_checkout = false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    project = tmp_path / "project"
    local_cfg = project / ".lelabo" / "config.toml"
    local_cfg.parent.mkdir(parents=True, exist_ok=True)
    local_cfg.write_text(
        '\n'.join(
            [
                "[github]",
                'owner = "local-owner"',
                "",
                "[capsules]",
                'default_checkout_dir = "workspace"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(project)
    rc = config_cli.main(["show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert 'owner = "local-owner"' in out
    assert 'default_checkout_dir = "workspace"' in out
    assert 'default_branch = "main"' in out


def test_config_cli_path_and_set_use_human_blocks(tmp_path, capsys) -> None:
    cfg = tmp_path / "settings.toml"

    rc = config_cli.main(["path"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Config paths" in out
    assert "global:" in out
    assert "local:" in out

    rc = config_cli.main(["set", "github.owner", "acme", "--file", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Success: Config updated." in out
    assert "Config change" in out
    assert "key: github.owner" in out
    assert "value: acme" in out
