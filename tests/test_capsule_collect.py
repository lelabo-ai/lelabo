from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
collect = importlib.import_module("lab.capsule.collect")


def test_collect_single_run(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(
        json.dumps({"argv": ["python", "-m", "lab.main", "--seed", "1"]}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text('{"t":"train"}\n', encoding="utf-8")

    stage = tmp_path / "stage"
    manifest = collect.collect_capsule(source=run_dir, stage_dir=stage, capsule_id="cap_test")

    assert manifest["kind"] == "single_run"
    assert (stage / "manifest.json").exists()
    assert (stage / "logs" / "meta.json").exists()
    assert (stage / "env" / "deps_snapshot.txt").exists()
    assert (stage / "code" / "git.json").exists()
