from __future__ import annotations

import importlib
import sys

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
checksums = importlib.import_module("lab.capsule.checksums")


def test_checksums_roundtrip(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world", encoding="utf-8")

    checksums.write_checksums(tmp_path)
    ok, errs = checksums.verify_checksums(tmp_path)
    assert ok
    assert not errs

    (tmp_path / "a.txt").write_text("HELLO", encoding="utf-8")
    ok2, errs2 = checksums.verify_checksums(tmp_path)
    assert not ok2
    assert errs2
