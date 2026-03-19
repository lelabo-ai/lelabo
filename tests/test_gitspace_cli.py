from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
gitspace_cli = importlib.import_module("lelabo.cli.commands.gitspace")
capsule_create = importlib.import_module("lelabo.capsule.create")


def test_gitspace_cli_init_show_list_and_add(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    rc = gitspace_cli.main(["init", str(workspace), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.gitspace/v1"
    assert payload["command"] == "init"
    assert (workspace / ".lelabo" / "gitspace.toml").exists()

    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )

    rc = gitspace_cli.main(["add", str(capsule_root), str(workspace), "--json"])
    assert rc == 0
    add_payload = json.loads(capsys.readouterr().out)
    assert add_payload["command"] == "add"
    assert add_payload["result"]["capsule"]["id"] == "demo_capsule"

    monkeypatch.chdir(workspace)
    rc = gitspace_cli.main(["show", "--json"])
    assert rc == 0
    show_payload = json.loads(capsys.readouterr().out)
    assert show_payload["gitspace"]["name"] == "workspace"

    rc = gitspace_cli.main(["list", "--json"])
    assert rc == 0
    list_payload = json.loads(capsys.readouterr().out)
    assert list_payload["schema_version"] == "lelabo.cli.gitspaces/v1"
    assert list_payload["capsules"] == [{"id": "demo_capsule", "path": "demo_capsule"}]


def test_gitspace_cli_rejects_invalid_capsule_path(tmp_path) -> None:
    workspace = tmp_path / "workspace_invalid"
    workspace.mkdir()
    rc = gitspace_cli.main(["init", str(workspace)])
    assert rc == 0

    try:
        gitspace_cli.main(["add", str(tmp_path / "missing_capsule"), str(workspace)])
    except SystemExit as exc:
        assert "Capsule source directory not found" in str(exc)
    else:
        raise AssertionError("Expected SystemExit when adding a missing capsule to a gitspace.")


def test_gitspace_cli_human_output_uses_consistent_blocks(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_human"
    workspace.mkdir()

    rc = gitspace_cli.main(["init", str(workspace)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Success: Gitspace initialized." in out
    assert "Gitspace" in out
    assert "manifest_path:" in out

    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )

    rc = gitspace_cli.main(["add", str(capsule_root), str(workspace)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Success: Capsule added to gitspace." in out
    assert "capsule_id: demo_capsule" in out

    monkeypatch.chdir(workspace)
    rc = gitspace_cli.main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Gitspace capsules" in out
    assert "demo_capsule | path: demo_capsule" in out
