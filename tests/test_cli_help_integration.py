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


def test_audit_help_displays_audit_parser_usage() -> None:
    proc = _run_cli_help("audit", "-h")
    assert proc.returncode == 0
    assert "usage: lelabo audit" in proc.stdout
    assert "Run warn-only local update-rule audit" in proc.stdout
