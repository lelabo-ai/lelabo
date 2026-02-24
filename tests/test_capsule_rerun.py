from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
pack = importlib.import_module("lelabo.capsule.pack")
install = importlib.import_module("lelabo.capsule.install")
rerun = importlib.import_module("lelabo.capsule.rerun")


def test_rerun_capsule_current_env(tmp_path) -> None:
    marker = tmp_path / "rerun_marker.txt"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cmd = [sys.executable, "-c", f"from pathlib import Path; Path(r'{marker}').write_text('ok', encoding='utf-8')"]
    (run_dir / "meta.json").write_text(json.dumps({"argv": cmd}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    bundle = pack.pack_capsule(source=run_dir, out_path=tmp_path / "capsule.tar.gz", capsule_id="cap_rerun")
    caps_dir = tmp_path / "caps"
    install.install_capsule(bundle_path=bundle, alias="baseline_rerun", capsules_dir=caps_dir)

    code = rerun.rerun_capsule(capsule_or_alias="baseline_rerun", capsules_dir=caps_dir, env_mode="current")
    assert code == 0
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "ok"
