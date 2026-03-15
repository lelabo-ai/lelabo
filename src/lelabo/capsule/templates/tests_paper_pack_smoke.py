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
MODEL_FILE = CAPSULE_ROOT / "models" / "example.py"
RULE_FILE = CAPSULE_ROOT / "update_rules" / "example.py"
CONFIG_PATH = CAPSULE_ROOT / "configs" / "train.supervised.paper_pack.toml"


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


def _paper_pack_enabled() -> bool:
    model_text = MODEL_FILE.read_text(encoding="utf-8")
    rule_text = RULE_FILE.read_text(encoding="utf-8")
    model_enabled = re.search(r'(?m)^\s*@register_model\("example_mlp"\)\s*$', model_text) is not None
    rule_enabled = re.search(r'(?m)^\s*@register_update_rule\("local_head"\)\s*$', rule_text) is not None
    return bool(model_enabled and rule_enabled)


def test_paper_pack_smoke(tmp_path: Path) -> None:
    if not _paper_pack_enabled():
        pytest.skip(
            'Uncomment `@register_model("example_mlp")` in `models/example.py` and '
            'Uncomment `@register_update_rule("local_head")` in `update_rules/example.py` '
            "to enable this smoke test."
        )

    listed_models = _run("list", "models", "--json")
    assert listed_models.returncode == 0, listed_models.stderr or listed_models.stdout
    assert "example_mlp" in json.loads(listed_models.stdout)["sources"]["capsule"]

    listed_rules = _run("list", "update-rules", "--json")
    assert listed_rules.returncode == 0, listed_rules.stderr or listed_rules.stdout
    assert "local_head" in json.loads(listed_rules.stdout)["sources"]["capsule"]

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
        str(tmp_path / "paper_pack_smoke"),
    )
    assert trained.returncode == 0, trained.stderr or trained.stdout
