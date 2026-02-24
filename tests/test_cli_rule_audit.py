from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
cli_main = importlib.import_module("lelabo.cli.main")
audit_cli = importlib.import_module("lelabo.cli.commands.audit")
api_audit = importlib.import_module("lelabo.api.audit")


def test_lelabo_audit_rule_invokes_pytest_with_expected_env(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(api_audit.subprocess, "call", _fake_call)

    rc = audit_cli.main(["bp"])
    assert rc == 0
    assert len(calls) == 1

    cmd, env = calls[0]
    assert cmd[0].endswith("python")
    assert cmd[1:4] == ["-m", "pytest", "-q"]
    assert "heavy" in cmd
    assert "tests/test_update_rules_research_audit.py" in cmd

    assert env is not None
    assert env["LELABO_RULE_AUDIT"] == "1"
    assert env["LELABO_RULE_AUDIT_ALGOS"] == "bp"
    assert env["LELABO_RULE_AUDIT_MODES"] == "supervised,rl"
    assert env["LELABO_RULE_AUDIT_MODEL"] == "auto"
    assert env["LELABO_RULE_AUDIT_EPOCHS"] == "1"
    assert env["LELABO_RULE_AUDIT_STEPS_PER_EPOCH"] == "1"


def test_lelabo_audit_all_does_not_force_specific_algo(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(api_audit.subprocess, "call", _fake_call)

    rc = audit_cli.main(["--all", "--modes", "supervised"])
    assert rc == 0
    assert len(calls) == 1

    _, env = calls[0]
    assert env is not None
    assert env["LELABO_RULE_AUDIT"] == "1"
    assert env["LELABO_RULE_AUDIT_MODES"] == "supervised"
    assert "LELABO_RULE_AUDIT_ALGOS" not in env


def test_lelabo_audit_accepts_model_and_epoch_options(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(api_audit.subprocess, "call", _fake_call)

    rc = audit_cli.main(["softhebb", "--model", "deephebb", "--epochs", "3", "--steps-per-epoch", "2"])
    assert rc == 0
    assert len(calls) == 1

    _, env = calls[0]
    assert env is not None
    assert env["LELABO_RULE_AUDIT_ALGOS"] == "softhebb"
    assert env["LELABO_RULE_AUDIT_MODEL"] == "deephebb"
    assert env["LELABO_RULE_AUDIT_EPOCHS"] == "3"
    assert env["LELABO_RULE_AUDIT_STEPS_PER_EPOCH"] == "2"


def test_lelabo_audit_show_warnings_and_extra_pytest_args(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(api_audit.subprocess, "call", _fake_call)

    rc = audit_cli.main(["bp", "--show-warnings", "--", "-s"])
    assert rc == 0
    assert len(calls) == 1

    cmd, _ = calls[0]
    assert "-W" in cmd
    assert "default" in cmd
    assert "-s" in cmd


def test_lelabo_audit_accepts_multiple_models(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(api_audit.subprocess, "call", _fake_call)

    rc = audit_cli.main(["bp", "--model", "mlp,cnn,transformer", "--epochs", "2"])
    assert rc == 0
    assert len(calls) == 1

    _, env = calls[0]
    assert env is not None
    assert env["LELABO_RULE_AUDIT_MODEL"] == "mlp,cnn,transformer"
    assert env["LELABO_RULE_AUDIT_EPOCHS"] == "2"


def test_root_help_prints_and_returns_zero(capsys) -> None:
    rc = cli_main.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LeLabo command-line interface" in out
    assert "train" in out
    assert "audit" in out
    assert "create" in out
    assert "list" in out
    assert "capsule" in out


def test_missing_command_with_args_is_error() -> None:
    with pytest.raises(SystemExit):
        cli_main.main(["--dataset", "iris"])


def test_test_command_is_removed() -> None:
    with pytest.raises(SystemExit):
        cli_main.main(["test", "bp"])


def test_train_dispatch_forwards_arguments(monkeypatch) -> None:
    seen = {}

    def _fake_run_train(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_main, "_run_train_cli", _fake_run_train)
    rc = cli_main.main(["train", "supervised", "--dataset", "iris", "--seed", "7"])
    assert rc == 0
    assert seen["argv"] == ["supervised", "--dataset", "iris", "--seed", "7"]


def test_train_help_is_dispatched_to_train_cli(monkeypatch) -> None:
    seen = {}

    def _fake_run_train(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_main, "_run_train_cli", _fake_run_train)
    rc = cli_main.main(["train", "-h"])
    assert rc == 0
    assert seen["argv"] == ["-h"]


def test_audit_help_is_dispatched_to_audit_cli(monkeypatch) -> None:
    seen = {}

    def _fake_run_audit(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_main, "_run_audit_cli", _fake_run_audit)
    rc = cli_main.main(["audit", "-h"])
    assert rc == 0
    assert seen["argv"] == ["-h"]


def test_create_help_is_dispatched_to_create_cli(monkeypatch) -> None:
    seen = {}

    def _fake_run_create(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_main, "_run_create_cli", _fake_run_create)
    rc = cli_main.main(["create", "-h"])
    assert rc == 0
    assert seen["argv"] == ["--help"]


def test_list_help_is_dispatched_to_list_cli(monkeypatch) -> None:
    seen = {}

    def _fake_run_list(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_main, "_run_list_cli", _fake_run_list)
    rc = cli_main.main(["list", "-h"])
    assert rc == 0
    assert seen["argv"] == ["--help"]


def test_capsule_help_is_dispatched_to_capsule_cli(monkeypatch) -> None:
    seen = {}

    def _fake_run_capsule(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_main, "_run_capsule_cli", _fake_run_capsule)
    rc = cli_main.main(["capsule", "-h"])
    assert rc == 0
    assert seen["argv"] == ["--help"]


def test_capsule_errors_return_nonzero_and_print_message(monkeypatch, capsys) -> None:
    def _fake_run_capsule(argv):
        raise SystemExit("Unknown capsule 'missing'")

    monkeypatch.setattr(cli_main, "run_capsule_command", _fake_run_capsule)
    rc = cli_main._run_capsule_cli(["show", "missing"])
    assert rc == 1
    assert "Unknown capsule 'missing'" in capsys.readouterr().err
