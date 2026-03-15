from __future__ import annotations

import subprocess
import sys


def _run_cli_help(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "lelabo.cli.main", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_train_help_displays_train_root_usage() -> None:
    proc = _run_cli_help("train", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo train" in proc.stdout
    assert "supervised" in proc.stdout
    assert "rl" in proc.stdout


def test_train_supervised_help_displays_parser_usage() -> None:
    proc = _run_cli_help("train", "supervised", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo train supervised" in proc.stdout
    assert "--dataset" in proc.stdout
    assert "Override dataset.name." in proc.stdout
    assert "Override model.name." in proc.stdout
    assert "--rule" in proc.stdout
    assert "Override update_rule.name." in proc.stdout
    assert "Override optimizer.name." in proc.stdout
    assert "Override train.epochs." in proc.stdout
    assert "Override train.batch." in proc.stdout
    assert "--task" not in proc.stdout


def test_train_rl_help_displays_parser_usage() -> None:
    proc = _run_cli_help("train", "rl", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo train rl" in proc.stdout
    assert "--env ENV_ID" in proc.stdout
    assert "--algo" in proc.stdout
    assert "--rule" in proc.stdout
    assert "--task" not in proc.stdout

def test_audit_help_displays_audit_parser_usage() -> None:
    proc = _run_cli_help("audit", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo audit" in proc.stdout
    assert "Run warn-only local update-rule audit" in proc.stdout

def test_list_help_displays_list_usage() -> None:
    proc = _run_cli_help("list", "-h")
    assert proc.returncode == 0
    assert "List available LeLabo registries" in proc.stdout
    assert "lelabo list [target] [--json]" in proc.stdout
    assert "update-rules" in proc.stdout
    assert "datasets" in proc.stdout
    assert "initializers" in proc.stdout
    assert "optimizers" in proc.stdout
    assert "losses" in proc.stdout
    assert "schedulers" in proc.stdout


def test_train_without_subcommand_prints_help() -> None:
    proc = _run_cli_help("train")
    assert proc.returncode == 0
    assert "usage: lelabo train" in proc.stdout


def test_train_supervised_requires_dataset_flag() -> None:
    proc = _run_cli_help("train", "supervised")
    assert proc.returncode != 0
    assert "--dataset" in proc.stderr


def test_train_rl_requires_env_flag() -> None:
    proc = _run_cli_help("train", "rl")
    assert proc.returncode != 0
    assert "--env" in proc.stderr


def test_root_cli_rejects_removed_create_command() -> None:
    proc = _run_cli_help("create", "-h")
    assert proc.returncode != 0
    assert "Unknown command: create" in proc.stderr


def test_train_supervised_rejects_legacy_source_flag() -> None:
    proc = _run_cli_help("train", "supervised", "--dataset", "iris", "--source", "iris")
    assert proc.returncode != 0
    assert "unrecognized arguments: --source iris" in proc.stderr


def test_train_rl_rejects_legacy_rl_specific_flag() -> None:
    proc = _run_cli_help("train", "rl", "--env", "CartPole-v1", "--gamma", "0.95")
    assert proc.returncode != 0
    assert "unrecognized arguments: --gamma 0.95" in proc.stderr
