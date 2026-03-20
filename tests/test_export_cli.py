from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
export_cli = importlib.import_module("lelabo.cli.commands.export")
capsule_create = importlib.import_module("lelabo.capsule.create")


def test_export_cli_exports_active_capsule_bundle(tmp_path, monkeypatch, capsys) -> None:
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="export_capsule",
        base_dir=tmp_path,
        register=False,
    )
    monkeypatch.chdir(capsule_root)

    rc = export_cli.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.export/v1"
    assert payload["command"] == "export"
    bundle_path = Path(payload["result"]["bundle_path"])
    assert bundle_path.name == "export_capsule.tar.gz"
    assert bundle_path.exists()


def test_export_cli_uses_picker_for_multiple_workspace_capsules(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_export"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=workspace, register=False)
    cap_b = capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=workspace, register=False)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(export_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(export_cli, "pick_many_with_checkboxes", lambda **kwargs: ["cap_b"])

    rc = export_cli.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "cap_b"
    assert Path(payload["result"]["bundle_path"]).exists()
    assert payload["capsule"]["path"] == str(cap_b)
