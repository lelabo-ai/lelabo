"""Template smoke test for the capsule optimizer golden path."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


CAPSULE_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_FILE = CAPSULE_ROOT / "optimizers" / "example.py"
CONFIG_PATH = CAPSULE_ROOT / "configs" / "train" / "supervised.capsule_optimizer.toml"


def _lelabo_cmd() -> list[str]:
    exe = shutil.which("lelabo")
    if exe:
        return [exe]
    return [sys.executable, "-m", "lelabo.cli.main"]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    return subprocess.run(
        [*_lelabo_cmd(), *args],
        cwd=CAPSULE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _capsule_optimizer_enabled() -> bool:
    text = OPTIMIZER_FILE.read_text(encoding="utf-8")
    return re.search(r'(?m)^\s*@register_optimizer\("capsule_sgd"\)\s*$', text) is not None


def test_capsule_optimizer_smoke(tmp_path: Path) -> None:
    if not _capsule_optimizer_enabled():
        pytest.skip(
            'Uncomment `@register_optimizer("capsule_sgd")` in `optimizers/example.py` '
            "to enable this smoke test."
        )

    listed = _run("list", "optimizers", "--json")
    assert listed.returncode == 0, listed.stderr or listed.stdout
    payload = json.loads(listed.stdout)
    assert "capsule_sgd" in payload.get("sources", {}).get("capsule", [])

    trained = _run(
        "train",
        "supervised",
        "--config",
        str(CONFIG_PATH),
        "--epochs",
        "1",
        "--display",
        "none",
        "--run-dir",
        str(tmp_path / "capsule_optimizer_smoke"),
    )
    assert trained.returncode == 0, trained.stderr or trained.stdout
"""Template smoke test for the capsule optimizer golden path."""
