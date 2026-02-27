from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
collect = importlib.import_module("lelabo.capsule.collect")


def test_collect_sweep(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "plan.json").write_text(json.dumps([{"idx": 0}]), encoding="utf-8")

    r1 = sweep / "job1"
    r1.mkdir()
    (r1 / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo", "--seed", "1"]}), encoding="utf-8")
    (r1 / "summary.json").write_text(json.dumps({"acc": 0.5}), encoding="utf-8")

    stage = tmp_path / "stage"
    manifest = collect.collect_capsule(source=sweep, stage_dir=stage, capsule_id="sweep_cap")

    assert manifest["kind"] == "sweep"
    assert (stage / "plan.json").exists()
    assert (stage / "runs" / "job1" / "summary.json").exists()
