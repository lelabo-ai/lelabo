from __future__ import annotations

import importlib
import sys

import pytest

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

    removed = registry.remove_capsule_entry("baseline1", capsules_dir)
    assert removed["capsule_id"] == "cap1"
    assert removed["aliases"] == ["baseline1"]
    assert registry.get_capsule("baseline1", capsules_dir) is None
    assert registry.list_capsules(capsules_dir) == []


def test_default_capsules_dir_resolves_parent_index(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    capsules_dir = project_root / ".lelabo" / "capsules"
    cap_dir = project_root / "capsule_a"
    cap_dir.mkdir(parents=True)

    monkeypatch.chdir(project_root)
    registry.add_capsule_entry(
        capsule_id="cap_parent",
        capsule_path=cap_dir,
        manifest={"kind": "config_only", "created_at": "2026-02-20T00:00:00Z", "source": {"path": "x"}},
        alias="cap_parent",
        capsules_dir=capsules_dir,
    )

    nested = cap_dir / "models"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert registry.default_capsules_dir() == capsules_dir.resolve()


def test_default_capsules_dir_uses_user_data_when_no_legacy_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LELABO_CAPSULES_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg_data"))
    monkeypatch.chdir(tmp_path)
    assert registry.default_capsules_dir() == (tmp_path / "xdg_data" / "lelabo" / "capsules").resolve()


def test_registry_load_index_recovers_from_corrupted_json(tmp_path) -> None:
    capsules_dir = tmp_path / "capsules"
    idx_path = registry.index_path(capsules_dir)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text("{not json", encoding="utf-8")

    loaded = registry.load_index(capsules_dir)
    assert loaded["capsules"] == {}
    assert loaded["aliases"] == {}
    backups = list(idx_path.parent.glob("index.json.corrupt.*.bak"))
    assert backups
    assert not idx_path.exists()


def test_registry_rejects_alias_collision(tmp_path) -> None:
    capsules_dir = tmp_path / "capsules"
    cap1 = capsules_dir / "cap1"
    cap2 = capsules_dir / "cap2"
    cap1.mkdir(parents=True)
    cap2.mkdir(parents=True)
    (cap1 / "manifest.json").write_text("{}", encoding="utf-8")
    (cap2 / "manifest.json").write_text("{}", encoding="utf-8")

    registry.add_capsule_entry(
        capsule_id="cap1",
        capsule_path=cap1,
        manifest={"kind": "config_only", "created_at": "2026-02-20T00:00:00Z", "source": {"path": "x"}},
        alias="baseline",
        capsules_dir=capsules_dir,
    )

    with pytest.raises(ValueError, match="Alias 'baseline' is already used"):
        registry.add_capsule_entry(
            capsule_id="cap2",
            capsule_path=cap2,
            manifest={"kind": "config_only", "created_at": "2026-02-20T00:00:00Z", "source": {"path": "x"}},
            alias="baseline",
            capsules_dir=capsules_dir,
        )
