from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
create_cli = importlib.import_module("lab.cli.commands.create")
capsule_cli = importlib.import_module("lab.cli.commands.capsule")


def test_create_cli_capsule_creates_layout(tmp_path, capsys) -> None:
    caps_dir = tmp_path / ".lelabo" / "capsules"
    rc = create_cli.main(
        ["capsule", "--name", "cli_capsule", "--dir", str(tmp_path), "--capsules-dir", str(caps_dir)]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "cli_capsule" in out
    assert (tmp_path / "cli_capsule" / "models").is_dir()
    assert (tmp_path / "cli_capsule" / "update_rules").is_dir()
    assert (tmp_path / "cli_capsule" / "metrics").is_dir()
    assert (tmp_path / "cli_capsule" / "optimizers").is_dir()
    assert (tmp_path / "cli_capsule" / "models" / "example.py").exists()

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out2 = capsys.readouterr().out
    assert "cli_capsule" in out2


def test_create_cli_capsule_accepts_positional_name(tmp_path) -> None:
    rc = create_cli.main(
        ["capsule", "positional_capsule", "--dir", str(tmp_path), "--capsules-dir", str(tmp_path / ".lelabo" / "capsules")]
    )
    assert rc == 0
    assert (tmp_path / "positional_capsule" / "datasets").is_dir()


def test_create_cli_unknown_target_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        create_cli.main(["unknown_target"])


def test_create_cli_can_skip_registry_registration(tmp_path) -> None:
    rc = create_cli.main(["capsule", "--name", "local_only", "--dir", str(tmp_path), "--no-register"])
    assert rc == 0
    rows = capsule_cli.list_capsules(tmp_path / ".lelabo" / "capsules")
    assert not rows
