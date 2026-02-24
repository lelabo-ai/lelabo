from __future__ import annotations

import importlib
import json
import sys
import tarfile

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
pack = importlib.import_module("lelabo.capsule.pack")


def test_pack_capsule_creates_bundle(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo.main"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.9}), encoding="utf-8")

    out = tmp_path / "demo_capsule.tar.gz"
    got = pack.pack_capsule(source=run_dir, out_path=out, capsule_id="demo_capsule")

    assert got == out
    assert out.exists()

    with tarfile.open(out, "r:gz") as tf:
        names = set(tf.getnames())
    assert "manifest.json" in names
    assert "checksums.sha256" in names
