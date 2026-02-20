from __future__ import annotations

import importlib
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
registry = importlib.import_module("lab.capsule.registry")


def test_registry_add_and_resolve(tmp_path) -> None:
    capsules_dir = tmp_path / "capsules"
    cap_dir = capsules_dir / "cap1"
    cap_dir.mkdir(parents=True)
    (cap_dir / "manifest.json").write_text("{}", encoding="utf-8")

    entry = registry.add_capsule_entry(
        capsule_id="cap1",
        capsule_path=cap_dir,
        manifest={"kind": "single_run", "created_at": "2026-02-20T00:00:00Z", "source": {"path": "x"}},
        alias="baseline1",
        capsules_dir=capsules_dir,
        source_bundle="/tmp/bundle.tar.gz",
    )
    assert entry["capsule_id"] == "cap1"

    got = registry.get_capsule("baseline1", capsules_dir)
    assert got is not None
    assert got["capsule_id"] == "cap1"

    rows = registry.list_capsules(capsules_dir)
    assert len(rows) == 1
    assert rows[0]["aliases"] == ["baseline1"]
