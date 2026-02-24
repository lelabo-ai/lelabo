from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
pack = importlib.import_module("lelabo.capsule.pack")
install = importlib.import_module("lelabo.capsule.install")
registry = importlib.import_module("lelabo.capsule.registry")


def test_install_capsule_updates_registry(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo.main"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.9}), encoding="utf-8")

    bundle = pack.pack_capsule(source=run_dir, out_path=tmp_path / "capsule.tar.gz", capsule_id="cap_install")
    capsules_dir = tmp_path / "caps_store"
    entry = install.install_capsule(bundle_path=bundle, alias="baseline_install", capsules_dir=capsules_dir)

    assert entry["capsule_id"] == "cap_install"
    got = registry.get_capsule("baseline_install", capsules_dir)
    assert got is not None
    assert (capsules_dir / "cap_install" / "manifest.json").exists()
