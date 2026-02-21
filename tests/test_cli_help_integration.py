from __future__ import annotations

import subprocess
import sys


def _run_cli_help(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "lab.cli.main", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_train_help_displays_train_parser_usage() -> None:
    proc = _run_cli_help("train", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo train" in proc.stdout
    assert "Run a LeLabo experiment" in proc.stdout
    assert "--source SOURCE" in proc.stdout
    assert "--task {auto,supervised,rl}" in proc.stdout

def test_audit_help_displays_audit_parser_usage() -> None:
    proc = _run_cli_help("audit", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo audit" in proc.stdout
    assert "Run warn-only local update-rule audit" in proc.stdout


def test_train_requires_source_flag() -> None:
    proc = _run_cli_help("train")
    assert proc.returncode != 0
    assert "--source" in proc.stderr


def test_train_rejects_legacy_dataset_flag() -> None:
    proc = _run_cli_help("train", "--source", "iris", "--dataset", "mnist")
    assert proc.returncode != 0
    assert "unrecognized arguments: --dataset mnist" in proc.stderr


def test_train_rejects_legacy_rl_specific_flag() -> None:
    proc = _run_cli_help("train", "--source", "CartPole-v1", "--task", "rl", "--gamma", "0.95")
    assert proc.returncode != 0
    assert "unrecognized arguments: --gamma 0.95" in proc.stderr
