from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
schema = importlib.import_module("lab.capsule.schema")


def test_validate_manifest_ok() -> None:
    manifest = {
        "schema_version": schema.CAPSULE_SCHEMA_VERSION,
        "capsule_id": "demo",
        "created_at": "2026-02-20T12:00:00Z",
        "kind": "single_run",
        "source": {"path": "/tmp/x", "type": "directory"},
        "entrypoints": [{"name": "run", "cmd": ["python", "-m", "lab.main"]}],
        "artifacts": {},
        "replay": {"command": ["python", "-m", "lab.main"]},
    }
    out = schema.validate_manifest(manifest)
    assert out["capsule_id"] == "demo"


def test_validate_manifest_missing_keys() -> None:
    with pytest.raises(Exception):
        schema.validate_manifest({"schema_version": schema.CAPSULE_SCHEMA_VERSION})
