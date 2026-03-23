from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
export_cli = importlib.import_module("lelabo.cli.commands.export")
capsule_cli = importlib.import_module("lelabo.cli.commands.capsule")
capsule_create = importlib.import_module("lelabo.capsule.create")


def test_export_cli_exports_visible_workspace_capsule_bundle(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_export"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="export_capsule",
        base_dir=workspace,
        register=False,
    )
    monkeypatch.chdir(workspace)

    rc = export_cli.main(["export_capsule", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.export/v1"
    assert payload["command"] == "export"
    assert payload["capsule"]["capsule_id"] == "export_capsule"
    assert payload["capsule"]["status"] == "workspace"
    bundle_path = Path(payload["result"]["bundle_path"])
    assert bundle_path.name == "export_capsule.tar.gz"
    assert bundle_path.exists()


def test_export_cli_visible_workspace_capsule_roundtrips_with_capsule_list(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_roundtrip"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    monkeypatch.chdir(workspace)

    rc = capsule_cli.main(["list", "--json"])
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    assert [row["capsule_id"] for row in listed["capsules"]] == ["demo_capsule"]

    rc = export_cli.main(["demo_capsule", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "demo_capsule"


def test_export_cli_accepts_local_path(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_path_export"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    monkeypatch.chdir(workspace)

    rc = export_cli.main([f"./{capsule_root.name}", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "demo_capsule"
    assert payload["capsule"]["path"] == str(capsule_root.resolve())


def test_export_cli_unknown_capsule_is_guided(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_unknown_export"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    with pytest.raises(SystemExit) as exc:
        export_cli.main(["missing_capsule"])

    assert "Unknown capsule 'missing_capsule'." in str(exc.value)
    assert "`lelabo capsule list`" in str(exc.value)
