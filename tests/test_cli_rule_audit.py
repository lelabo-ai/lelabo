from __future__ import annotations

from conftest import REPO_ROOT, load_module_from_path


cli_main = load_module_from_path(
    "lelabo_cli_main_test_module",
    REPO_ROOT / "src" / "lab" / "cli" / "main.py",
)


def test_lelabo_test_rule_invokes_pytest_with_expected_env(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(cli_main.subprocess, "call", _fake_call)

    rc = cli_main.main(["test", "bp"])
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


def test_lelabo_test_all_does_not_force_specific_algo(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(cli_main.subprocess, "call", _fake_call)

    rc = cli_main.main(["test", "--all", "--modes", "supervised"])
    assert rc == 0
    assert len(calls) == 1

    _, env = calls[0]
    assert env is not None
    assert env["LELABO_RULE_AUDIT"] == "1"
    assert env["LELABO_RULE_AUDIT_MODES"] == "supervised"
    assert "LELABO_RULE_AUDIT_ALGOS" not in env


def test_lelabo_test_accepts_model_and_epoch_options(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(cli_main.subprocess, "call", _fake_call)

    rc = cli_main.main(["test", "softhebb", "--model", "deephebb", "--epochs", "3", "--steps-per-epoch", "2"])
    assert rc == 0
    assert len(calls) == 1

    _, env = calls[0]
    assert env is not None
    assert env["LELABO_RULE_AUDIT_ALGOS"] == "softhebb"
    assert env["LELABO_RULE_AUDIT_MODEL"] == "deephebb"
    assert env["LELABO_RULE_AUDIT_EPOCHS"] == "3"
    assert env["LELABO_RULE_AUDIT_STEPS_PER_EPOCH"] == "2"


def test_lelabo_test_show_warnings_and_extra_pytest_args(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(cli_main.subprocess, "call", _fake_call)

    rc = cli_main.main(["test", "bp", "--show-warnings", "--", "-s"])
    assert rc == 0
    assert len(calls) == 1

    cmd, _ = calls[0]
    assert "-W" in cmd
    assert "default" in cmd
    assert "-s" in cmd


def test_lelabo_test_accepts_multiple_models(monkeypatch) -> None:
    calls = []

    def _fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(cli_main.subprocess, "call", _fake_call)

    rc = cli_main.main(["test", "bp", "--model", "mlp,cnn,transformer", "--epochs", "2"])
    assert rc == 0
    assert len(calls) == 1

    _, env = calls[0]
    assert env is not None
    assert env["LELABO_RULE_AUDIT_MODEL"] == "mlp,cnn,transformer"
    assert env["LELABO_RULE_AUDIT_EPOCHS"] == "2"
