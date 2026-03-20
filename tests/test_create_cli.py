from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
capsule_cli = importlib.import_module("lelabo.cli.commands.capsule")


def test_capsule_init_creates_local_layout_without_storing(tmp_path, capsys) -> None:
    rc = capsule_cli.main(["init", "cli_capsule", "--dir", str(tmp_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Success: Capsule initialized." in out
    assert "Capsule" in out
    assert "name: cli_capsule" in out
    assert f"path: {tmp_path / 'cli_capsule'}" in out
    assert (tmp_path / "cli_capsule" / "models").is_dir()
    assert (tmp_path / "cli_capsule" / "update_rules").is_dir()
    assert (tmp_path / "cli_capsule" / "metrics").is_dir()
    assert (tmp_path / "cli_capsule" / "optimizers").is_dir()
    assert (tmp_path / "cli_capsule" / "models" / "example.py").exists()


def test_capsule_init_unknown_subcommand_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        capsule_cli.main(["unknown_target"])


def test_capsule_init_does_not_register_in_capsule_store(tmp_path, capsys) -> None:
    caps_dir = tmp_path / ".lelabo" / "capsules"
    rc = capsule_cli.main(["init", "local_only", "--dir", str(tmp_path)])
    assert rc == 0
    capsys.readouterr()

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    assert "Info: The capsule store is empty." in capsys.readouterr().out
