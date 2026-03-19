from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
push_cli = importlib.import_module("lelabo.cli.commands.push")
repo_cli = importlib.import_module("lelabo.cli.commands.repo")
capsule_create = importlib.import_module("lelabo.capsule.create")


def test_push_cli_uses_workspace_target_when_available(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_push"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )

    monkeypatch.setattr(
        push_cli,
        "available_targets",
        lambda _: [
            {
                "name": "workspace",
                "kind": "workspace",
                "owner": "acme",
                "repo": "research",
                "branch": "main",
                "path": "demo_capsule",
            }
        ],
    )
    monkeypatch.setattr(
        push_cli,
        "push_workspace_target",
        lambda **kwargs: {
            "target_name": "workspace",
            "target_kind": "workspace",
            "owner": "acme",
            "repo": "research",
            "branch": "main",
            "path": "demo_capsule",
            "commit_message": "Update demo_capsule",
            "committed": True,
            "pushed": True,
        },
    )

    rc = push_cli.main([str(capsule_root), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.push/v1"
    assert payload["result"]["target_kind"] == "workspace"
    assert payload["result"]["repo"] == "research"


def test_push_cli_bootstraps_github_target_with_yes(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_bootstrap"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    publish_state = tmp_path / "publish_targets.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))
    monkeypatch.setattr(push_cli, "available_targets", lambda _: [])
    monkeypatch.setattr(push_cli, "infer_workspace_target", lambda _: None)
    monkeypatch.setattr(push_cli, "current_github_login", lambda: "acme")
    monkeypatch.setattr(
        push_cli,
        "_effective_settings",
        lambda: {
            "github": {
                "owner": "acme",
                "default_visibility": "private",
                "default_branch": "main",
                "create_repo_if_missing": True,
            },
            "capsules": {"store_dir": "", "default_checkout_dir": ".", "install_checkout": False},
        },
    )
    monkeypatch.setattr(
        push_cli,
        "push_github_target",
        lambda **kwargs: {
            "target_name": "github",
            "target_kind": "github",
            "owner": "acme",
            "repo": "demo_capsule",
            "branch": "main",
            "path": "demo_capsule",
            "commit_message": "Update demo_capsule",
            "created_repo": True,
            "committed": True,
            "pushed": True,
        },
    )

    rc = push_cli.main([str(capsule_root), "--yes", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["target_kind"] == "github"
    state = json.loads(publish_state.read_text(encoding="utf-8"))
    entry = state["capsules"][str(capsule_root.resolve())]
    assert entry["default_target"] == "github"
    assert "github" in entry["targets"]


def test_repo_cli_add_list_use_remove_roundtrip(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_repo"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    publish_state = tmp_path / "publish_targets.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))
    monkeypatch.setattr(repo_cli, "current_github_login", lambda: "acme")
    monkeypatch.setattr(
        repo_cli,
        "_effective_settings",
        lambda: {
            "github": {
                "owner": "",
                "default_visibility": "private",
                "default_branch": "main",
                "create_repo_if_missing": True,
            },
            "capsules": {"store_dir": "", "default_checkout_dir": ".", "install_checkout": False},
        },
    )

    rc = repo_cli.main(["list", str(capsule_root), "--json"])
    assert rc == 0
    empty_payload = json.loads(capsys.readouterr().out)
    assert empty_payload["targets"] == []

    rc = repo_cli.main(["add", str(capsule_root), "--name", "public", "--owner", "acme", "--repo", "method-zoo", "--default", "--json"])
    assert rc == 0
    add_payload = json.loads(capsys.readouterr().out)
    assert add_payload["target"]["name"] == "public"
    assert add_payload["target"]["default"] is True

    rc = repo_cli.main(["list", str(capsule_root), "--json"])
    assert rc == 0
    list_payload = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in list_payload["targets"]] == ["public"]

    rc = repo_cli.main(["use", "public", str(capsule_root), "--json"])
    assert rc == 0
    use_payload = json.loads(capsys.readouterr().out)
    assert use_payload["target"]["default"] is True

    rc = repo_cli.main(["remove", "public", str(capsule_root), "--json"])
    assert rc == 0
    remove_payload = json.loads(capsys.readouterr().out)
    assert remove_payload["target"]["name"] == "public"
