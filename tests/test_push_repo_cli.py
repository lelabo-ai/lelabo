from __future__ import annotations

import importlib
import json
import sys

import pytest

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

    target = {
        "name": "workspace",
        "kind": "workspace",
        "owner": "acme",
        "repo": "research",
        "branch": "main",
        "path": "demo_capsule",
    }
    monkeypatch.setattr(push_cli, "available_targets", lambda _: [target])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: ("workspace", None))
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


def test_push_cli_resolves_single_child_capsule_from_workspace(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_child_resolution"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    captured: dict[str, object] = {}

    def _fake_push_capsule(**kwargs):
        captured.update(kwargs)
        return [
            {
                "target_name": "workspace",
                "target_kind": "workspace",
                "owner": "acme",
                "repo": "research",
                "branch": "main",
                "path": "demo_capsule",
                "commit_message": "Update demo_capsule",
                "committed": False,
                "pushed": False,
            }
        ]

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(push_cli, "push_capsule", _fake_push_capsule)

    rc = push_cli.main(["--json"])
    assert rc == 0
    assert captured["capsule_root"] == capsule_root
    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "demo_capsule"


def test_push_cli_errors_when_workspace_has_no_capsule(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_empty"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    with pytest.raises(SystemExit) as exc:
        push_cli.main([])

    assert "No capsule found in the current workspace." in str(exc.value)
    assert "create a capsule under this directory" in str(exc.value)


def test_push_cli_errors_when_workspace_has_multiple_child_capsules(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_multi"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=workspace, register=False)
    capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=workspace, register=False)
    monkeypatch.chdir(workspace)

    with pytest.raises(SystemExit) as exc:
        push_cli.main([])

    assert "Multiple capsules found in the current workspace:" in str(exc.value)
    assert "cap_a" in str(exc.value)
    assert "cap_b" in str(exc.value)


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
            "repo": "lelabo-capsules",
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
    assert payload["result"]["repo"] == "lelabo-capsules"
    state = json.loads(publish_state.read_text(encoding="utf-8"))
    entry = state["capsules"][str(capsule_root.resolve())]
    assert entry["default_target"] == "github"
    assert "github" in entry["targets"]


def test_push_cli_last_used_target_is_suggested_before_picker(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_last_used"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    workspace_target = {
        "name": "workspace",
        "kind": "workspace",
        "owner": "acme",
        "repo": "research",
        "branch": "main",
        "path": "demo_capsule",
    }
    github_target = {
        "name": "github",
        "kind": "github",
        "owner": "acme",
        "repo": "lelabo-capsules",
        "branch": "main",
        "path": "demo_capsule",
        "visibility": "private",
    }
    prompts: list[str] = []

    monkeypatch.setattr(push_cli, "available_targets", lambda _: [workspace_target, github_target])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, "github"))
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(
        push_cli,
        "_confirm",
        lambda question, default=False: prompts.append(question) or False,
    )
    monkeypatch.setattr(push_cli, "pick_many_with_checkboxes", lambda **kwargs: ["workspace"])
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
            "committed": False,
            "pushed": True,
        },
    )

    rc = push_cli.main([str(capsule_root), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["target_name"] == "workspace"
    assert prompts
    assert prompts[0].startswith("Last used publish target:")


def test_push_cli_single_non_default_target_uses_picker(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_single_picker"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    target = {
        "name": "workspace",
        "kind": "workspace",
        "owner": "acme",
        "repo": "research",
        "branch": "main",
        "path": "demo_capsule",
    }
    calls: list[str] = []

    monkeypatch.setattr(push_cli, "available_targets", lambda _: [target])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, None))
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(
        push_cli,
        "pick_many_with_checkboxes",
        lambda **kwargs: calls.append(kwargs["title"]) or ["workspace"],
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
            "committed": False,
            "pushed": True,
        },
    )

    rc = push_cli.main([str(capsule_root), "--json"])
    assert rc == 0
    assert calls == ["Select publish targets"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["target_name"] == "workspace"


def test_push_cli_picker_can_select_multiple_targets(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_multi_targets"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    workspace_target = {
        "name": "workspace",
        "kind": "workspace",
        "owner": "acme",
        "repo": "research",
        "branch": "main",
        "path": "demo_capsule",
    }
    github_target = {
        "name": "github",
        "kind": "github",
        "owner": "acme",
        "repo": "lelabo-capsules",
        "branch": "main",
        "path": "demo_capsule",
        "visibility": "private",
    }

    monkeypatch.setattr(push_cli, "available_targets", lambda _: [workspace_target, github_target])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, None))
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(push_cli, "pick_many_with_checkboxes", lambda **kwargs: ["workspace", "github"])
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
            "committed": False,
            "pushed": True,
        },
    )
    monkeypatch.setattr(
        push_cli,
        "push_github_target",
        lambda **kwargs: {
            "target_name": "github",
            "target_kind": "github",
            "owner": "acme",
            "repo": "lelabo-capsules",
            "branch": "main",
            "path": "demo_capsule",
            "commit_message": "Update demo_capsule",
            "created_repo": False,
            "committed": False,
            "pushed": True,
        },
    )

    rc = push_cli.main([str(capsule_root), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.pushes/v1"
    assert [row["target_name"] for row in payload["results"]] == ["workspace", "github"]


def test_push_cli_picker_can_create_new_target_and_return_to_selection(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_create_target"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    workspace_target = {
        "name": "workspace",
        "kind": "workspace",
        "owner": "acme",
        "repo": "research",
        "branch": "main",
        "path": "demo_capsule",
    }
    selections = [
        [push_cli._CREATE_TARGET_OPTION],
        ["github"],
    ]

    monkeypatch.setattr(push_cli, "available_targets", lambda _: [workspace_target])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, None))
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(push_cli, "pick_many_with_checkboxes", lambda **kwargs: selections.pop(0))
    monkeypatch.setattr(
        push_cli,
        "_prompt_custom_github_target",
        lambda **kwargs: {
            "name": "github",
            "kind": "github",
            "owner": "acme",
            "repo": "lelabo-capsules",
            "branch": "main",
            "path": "demo_capsule",
            "visibility": "private",
            "_ephemeral": True,
        },
    )
    monkeypatch.setattr(
        push_cli,
        "push_github_target",
        lambda **kwargs: {
            "target_name": "github",
            "target_kind": "github",
            "owner": "acme",
            "repo": "lelabo-capsules",
            "branch": "main",
            "path": "demo_capsule",
            "commit_message": "Update demo_capsule",
            "created_repo": False,
            "committed": False,
            "pushed": False,
        },
    )

    rc = push_cli.main([str(capsule_root), "--preview", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["target_name"] == "github"
    assert payload["result"]["repo"] == "lelabo-capsules"


def test_push_cli_preview_no_target_custom_flow_uses_folder_wording_without_saving(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_preview_custom"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    publish_state = tmp_path / "publish_targets_preview.json"
    answers = iter(["2", "", "", "", "papers/demo_capsule", "", ""])

    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))
    monkeypatch.setattr(push_cli, "available_targets", lambda _: [])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, None))
    monkeypatch.setattr(push_cli, "current_github_login", lambda: "acme")
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
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
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(
        push_cli,
        "push_github_target",
        lambda **kwargs: {
            "target_name": kwargs["target"]["name"],
            "target_kind": "github",
            "owner": kwargs["target"]["owner"],
            "repo": kwargs["target"]["repo"],
            "branch": kwargs["target"]["branch"],
            "path": kwargs["target"]["path"],
            "commit_message": "Update demo_capsule",
            "created_repo": False,
            "committed": False,
            "pushed": False,
        },
    )

    rc = push_cli.main([str(capsule_root), "--preview"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Info: No saved publish target for demo_capsule." in out
    assert "Proposed GitHub target: acme/lelabo-capsules" in out
    assert "Folder: papers/demo_capsule" in out
    assert "Target name" not in out
    assert "folder: papers/demo_capsule" in out
    assert "path:" not in out
    assert not publish_state.exists()


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
