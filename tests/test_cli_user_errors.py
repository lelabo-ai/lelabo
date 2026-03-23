from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
cli_main = importlib.import_module("lelabo.cli.main")


def test_root_cli_prints_type_error_from_train_command(monkeypatch, capsys) -> None:
    def _fake_run_train(argv):
        _ = argv
        raise TypeError("Optimizer builder 'capsule_bad_opt' must return torch.optim.Optimizer")

    monkeypatch.setattr(cli_main, "run_train_command", _fake_run_train)
    rc = cli_main._run_train_cli(["supervised", "--dataset", "iris"])
    assert rc == 1
    assert "capsule_bad_opt" in capsys.readouterr().err


def test_root_cli_prints_import_error_from_train_command(monkeypatch, capsys) -> None:
    def _fake_run_train(argv):
        _ = argv
        raise ImportError("Dataset 'glue' requires optional NLP dependencies.")

    monkeypatch.setattr(cli_main, "run_train_command", _fake_run_train)
    rc = cli_main._run_train_cli(["supervised", "--dataset", "glue"])
    assert rc == 1
    assert "optional NLP dependencies" in capsys.readouterr().err


def test_root_cli_prints_runtime_error_from_registries_command(monkeypatch, capsys) -> None:
    def _fake_run_registries(argv):
        _ = argv
        raise RuntimeError("Failed to load installed capsule plugins: capsule_internal_bug")

    monkeypatch.setattr(cli_main, "run_registries_command", _fake_run_registries)
    rc = cli_main._run_registries_cli(["models"])
    assert rc == 1
    assert "capsule_internal_bug" in capsys.readouterr().err


def test_root_registries_cli_invalid_capsule_plugin_is_message_not_traceback(tmp_path, capsys) -> None:
    capsule_root = tmp_path / "capsule_internal_bug"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_internal_bug\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "models" / "broken.py").write_text(
        "from lelabo.models.registry import register_model\n\n"
        "BROKEN = missing_name  # noqa: F821\n\n"
        "@register_model('capsule_broken_model')\n"
        "def build_capsule_broken_model(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    rc = cli_main.main(["registries", "models", "--capsule", str(capsule_root)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "capsule_internal_bug" in err
    assert "missing_name" in err
    assert "Traceback" not in err


def test_root_registries_cli_reports_register_import_error_cleanly(tmp_path, capsys) -> None:
    capsule_root = tmp_path / "capsule_missing_register"
    (capsule_root / "models").mkdir(parents=True)
    (capsule_root / "capsule.toml").write_text(
        "[capsule]\nname = \"capsule_missing_register\"\nformat = \"lelabo.capsule.scaffold.v1\"\n",
        encoding="utf-8",
    )
    (capsule_root / "models" / "broken.py").write_text(
        "@register_model('capsule_missing_register_model')\n"
        "def build_capsule_missing_register_model(ctx, args):\n"
        "    return None\n",
        encoding="utf-8",
    )

    rc = cli_main.main(["registries", "models", "--capsule", str(capsule_root)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "does not import 'register_model'" in err
    assert "Traceback" not in err
