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


def test_push_cli_rejects_workspace_target(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_push"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )

    with pytest.raises(SystemExit) as exc:
        push_cli.main([str(capsule_root), "--target", "workspace"])

    assert "does not support the workspace target" in str(exc.value)
    assert "lelabo export" in str(exc.value)


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
                "target_name": "github",
                "target_kind": "github",
                "owner": "acme",
                "repo": "lelabo-capsules",
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


def test_push_cli_uses_capsule_picker_when_workspace_has_multiple_capsules(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_multi_picker"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=workspace, register=False)
    cap_b = capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=workspace, register=False)
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
                "path": "cap_b",
                "commit_message": "Update cap_b",
                "committed": False,
                "pushed": False,
            }
        ]

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(push_cli, "pick_many_with_checkboxes", lambda **kwargs: [str(cap_b)])
    monkeypatch.setattr(push_cli, "push_capsule", _fake_push_capsule)

    rc = push_cli.main(["--json"])
    assert rc == 0
    assert captured["capsule_root"] == cap_b
    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "cap_b"


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


def test_push_cli_no_target_uses_repo_target_setup_flow(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_bootstrap_flow"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    monkeypatch.setattr(push_cli, "available_targets", lambda _: [])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, None))
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(
        push_cli,
        "configure_github_target_for_capsule",
        lambda *args, **kwargs: {
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


def test_push_cli_last_used_target_is_suggested_before_picker(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_last_used"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
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

    archive_target = {
        "name": "archive",
        "kind": "github",
        "owner": "lab",
        "repo": "team-capsules",
        "branch": "main",
        "path": "demo_capsule",
        "visibility": "private",
    }
    monkeypatch.setattr(push_cli, "available_targets", lambda _: [github_target, archive_target])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, "github"))
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(
        push_cli,
        "_confirm",
        lambda question, default=False: prompts.append(question) or False,
    )
    monkeypatch.setattr(push_cli, "pick_many_with_checkboxes", lambda **kwargs: ["archive"])
    monkeypatch.setattr(
        push_cli,
        "push_github_target",
        lambda **kwargs: {
            "target_name": "archive",
            "target_kind": "github",
            "owner": "lab",
            "repo": "team-capsules",
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
    assert payload["result"]["target_name"] == "archive"
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
        "name": "github",
        "kind": "github",
        "owner": "acme",
        "repo": "lelabo-capsules",
        "branch": "main",
        "path": "demo_capsule",
        "visibility": "private",
    }
    calls: list[str] = []

    monkeypatch.setattr(push_cli, "available_targets", lambda _: [target])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, None))
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(
        push_cli,
        "pick_many_with_checkboxes",
        lambda **kwargs: calls.append(kwargs["title"]) or ["github"],
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
            "committed": False,
            "pushed": True,
        },
    )

    rc = push_cli.main([str(capsule_root), "--json"])
    assert rc == 0
    assert calls == ["Select publish targets"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["target_name"] == "github"


def test_push_cli_picker_filters_out_workspace_targets(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_remote_only"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    seen_options: list[tuple[str, str]] = []
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
    monkeypatch.setattr(
        push_cli,
        "pick_many_with_checkboxes",
        lambda **kwargs: seen_options.extend(kwargs["options"]) or ["github"],
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
    assert [name for name, _ in seen_options] == ["github", push_cli._CREATE_TARGET_OPTION]
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["target_name"] == "github"


def test_push_cli_picker_can_select_multiple_targets(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_multi_targets"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    github_target = {
        "name": "github",
        "kind": "github",
        "owner": "acme",
        "repo": "lelabo-capsules",
        "branch": "main",
        "path": "demo_capsule",
        "visibility": "private",
    }
    archive_target = {
        "name": "archive",
        "kind": "github",
        "owner": "lab",
        "repo": "team-capsules",
        "branch": "main",
        "path": "papers/demo_capsule",
        "visibility": "private",
    }
    confirms: list[str] = []

    monkeypatch.setattr(push_cli, "available_targets", lambda _: [github_target, archive_target])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, None))
    monkeypatch.setattr(push_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(push_cli, "pick_many_with_checkboxes", lambda **kwargs: ["github", "archive"])
    monkeypatch.setattr(
        push_cli,
        "_confirm",
        lambda question, default=False: confirms.append(question) or True,
    )
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
            "pushed": True,
        },
    )

    rc = push_cli.main([str(capsule_root)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Publish targets" in out
    assert "github | repo: acme/lelabo-capsules | folder: demo_capsule" in out
    assert "github | repo: lab/team-capsules | folder: papers/demo_capsule" in out
    assert "Success: Pushed 2 targets." in out
    assert confirms == ["Continue?"]


def test_push_cli_picker_can_create_new_target_and_return_to_selection(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_create_target"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    existing_target = {
        "name": "public",
        "kind": "github",
        "owner": "acme",
        "repo": "method-zoo",
        "branch": "main",
        "path": "demo_capsule",
        "visibility": "public",
    }
    selections = [
        [push_cli._CREATE_TARGET_OPTION],
        ["github"],
    ]

    monkeypatch.setattr(push_cli, "available_targets", lambda _: [existing_target])
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
    monkeypatch.setattr(push_cli, "available_targets", lambda _: [])
    monkeypatch.setattr(push_cli, "get_target_preferences", lambda _: (None, None))
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
    monkeypatch.setattr(
        push_cli,
        "configure_github_target_for_capsule",
        lambda *args, **kwargs: {
            "name": "github",
            "kind": "github",
            "owner": "acme",
            "repo": "lelabo-capsules",
            "branch": "main",
            "path": "papers/demo_capsule",
            "visibility": "private",
            "_ephemeral": True,
        },
    )
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
    assert "folder: papers/demo_capsule" in out
    assert "path:" not in out


def test_repo_cli_create_shared_repo(tmp_path, monkeypatch, capsys) -> None:
    created: list[tuple[str, str, str]] = []
    publish_state = tmp_path / "publish_targets_create.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))

    monkeypatch.setattr(repo_cli, "current_github_login", lambda: "acme")
    monkeypatch.setattr(repo_cli, "create_repo", lambda owner, repo, visibility: created.append((owner, repo, visibility)))
    monkeypatch.setattr(repo_cli, "repo_exists", lambda owner, repo: False)
    monkeypatch.setattr(
        repo_cli,
        "_effective_settings",
        lambda: {
            "github": {
                "owner": "",
                "default_visibility": "private",
                "default_branch": "main",
            }
        },
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(repo_cli, "_confirm", lambda question, default=False: True)

    rc = repo_cli.main(["create", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create"
    assert payload["target"]["owner"] == "acme"
    assert payload["target"]["repo"] == "lelabo-capsules"
    assert created == [("acme", "lelabo-capsules", "private")]


def test_repo_cli_create_is_idempotent_when_target_already_configured(tmp_path, monkeypatch, capsys) -> None:
    publish_state = tmp_path / "publish_targets_create_existing.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))
    repo_cli.save_global_target(
        {"kind": "github", "owner": "acme", "repo": "lelabo-capsules", "branch": "main", "visibility": "private"}
    )
    monkeypatch.setattr(repo_cli, "current_github_login", lambda: "acme")
    monkeypatch.setattr(
        repo_cli,
        "_effective_settings",
        lambda: {
            "github": {
                "owner": "",
                "default_visibility": "private",
                "default_branch": "main",
            }
        },
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    rc = repo_cli.main(["create", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "unchanged"
    assert payload["target"]["repo"] == "lelabo-capsules"
    assert len(repo_cli.list_global_targets()) == 1


def test_repo_cli_create_registers_existing_remote_repo_without_creation(tmp_path, monkeypatch, capsys) -> None:
    publish_state = tmp_path / "publish_targets_create_register.json"
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
            }
        },
    )
    monkeypatch.setattr(
        repo_cli,
        "github_repo_metadata",
        lambda owner, repo: {
            "kind": "github",
            "owner": owner,
            "repo": repo,
            "branch": "main",
            "visibility": "public",
        },
    )
    monkeypatch.setattr(repo_cli, "create_repo", lambda owner, repo, visibility: pytest.fail("create_repo should not be called"))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    rc = repo_cli.main(["create", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "registered"
    assert payload["target"]["visibility"] == "public"
    assert len(repo_cli.list_global_targets()) == 1


def test_repo_cli_add_capsule_existing_repo_roundtrip(tmp_path, monkeypatch, capsys) -> None:
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
    repo_cli.save_global_target(
        {"kind": "github", "owner": "acme", "repo": "method-zoo", "branch": "main", "visibility": "private"}
    )
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
    selections = iter([[repo_cli._REPO_ACTION_EXISTING], ["acme/method-zoo"]])
    answers = iter(["", "", ""])
    monkeypatch.setattr(repo_cli, "pick_many_with_checkboxes", lambda **kwargs: next(selections))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(repo_cli, "_confirm", lambda question, default=False: True)

    rc = repo_cli.main(["list", str(capsule_root), "--json"])
    assert rc == 0
    empty_payload = json.loads(capsys.readouterr().out)
    assert empty_payload["targets"] == []

    rc = repo_cli.main(["attach", str(capsule_root), "--json"])
    assert rc == 0
    add_payload = json.loads(capsys.readouterr().out)
    assert add_payload["target"]["repo"] == "method-zoo"
    assert add_payload["target"]["owner"] == "acme"
    assert add_payload["target"]["default"] is True

    rc = repo_cli.main(["list", str(capsule_root), "--json"])
    assert rc == 0
    list_payload = json.loads(capsys.readouterr().out)
    assert [item["repo"] for item in list_payload["targets"]] == ["method-zoo"]
    assert list_payload["targets"][0]["capsules"] == ["demo_capsule"]

    edit_selections = iter([["default"], [f"{capsule_root.resolve()}::github"]])
    monkeypatch.setattr(repo_cli, "pick_many_with_checkboxes", lambda **kwargs: next(edit_selections))
    rc = repo_cli.main(["edit", "acme/method-zoo", "--json"])
    assert rc == 0
    edit_payload = json.loads(capsys.readouterr().out)
    assert edit_payload["command"] == "edit-default"
    assert edit_payload["target"]["default"] is True

    rc = repo_cli.main(["detach", "acme/method-zoo", str(capsule_root), "--json"])
    assert rc == 0
    detach_payload = json.loads(capsys.readouterr().out)
    assert detach_payload["command"] == "detach"
    assert detach_payload["target"]["name"] == "github"


def test_repo_cli_attach_single_capsule_prints_context_and_does_not_prompt_visibility_for_existing_target(
    tmp_path, monkeypatch, capsys
) -> None:
    workspace = tmp_path / "workspace_attach_context"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    publish_state = tmp_path / "publish_targets_attach_context.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))
    repo_cli.save_global_target(
        {"kind": "github", "owner": "acme", "repo": "method-zoo", "branch": "main", "visibility": "private"}
    )
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        repo_cli,
        "_effective_settings",
        lambda: {
            "github": {
                "owner": "acme",
                "default_visibility": "private",
                "default_branch": "main",
            },
            "capsules": {"store_dir": "", "default_checkout_dir": ".", "install_checkout": False},
        },
    )
    selections = iter([[repo_cli._REPO_ACTION_EXISTING], ["acme/method-zoo"]])
    prompts: list[str] = []

    monkeypatch.setattr(repo_cli, "pick_many_with_checkboxes", lambda **kwargs: next(selections))
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or "")
    monkeypatch.setattr(repo_cli, "_confirm", lambda question, default=False: True)

    rc = repo_cli.main(["attach"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Capsule" in out
    assert "name: demo_capsule" in out
    assert str(capsule_root.resolve()) in out
    assert not any(prompt.startswith("Visibility") for prompt in prompts)


def test_repo_cli_edit_import_capsule_from_target_repo(tmp_path, monkeypatch, capsys) -> None:
    publish_state = tmp_path / "publish_targets_import.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))
    repo_cli.save_global_target(
        {"kind": "github", "owner": "acme", "repo": "method-zoo", "branch": "main", "visibility": "private"}
    )
    monkeypatch.setattr(
        repo_cli,
        "inspect_target_repo_capsules",
        lambda target: [
            {
                "capsule_id": "colleague_capsule",
                "root": str((tmp_path / "remote_capsule").resolve()),
                "folder": "colleague_capsule",
            }
        ],
    )
    monkeypatch.setattr(
        repo_cli,
        "install_capsule_from_directory",
        lambda **kwargs: {
            "capsule_id": "colleague_capsule",
            "path": str((tmp_path / "store" / "colleague_capsule").resolve()),
            "install_action": "installed",
        },
    )
    monkeypatch.setattr(repo_cli, "pick_many_with_checkboxes", lambda **kwargs: ["import"])

    rc = repo_cli.main(["edit", "acme/method-zoo", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "edit-import"
    assert payload["target"]["imported_capsule_id"] == "colleague_capsule"


def test_repo_cli_edit_remove_capsule_from_target_repo_and_detach_locally(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_remove_remote"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    publish_state = tmp_path / "publish_targets_remove_remote.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))
    repo_cli.save_configured_target(
        capsule_root,
        {"name": "github", "kind": "github", "owner": "acme", "repo": "method-zoo", "branch": "main", "path": "demo_capsule", "visibility": "private"},
        make_default=True,
    )
    monkeypatch.setattr(repo_cli, "pick_many_with_checkboxes", lambda **kwargs: ["remove-remote"])
    monkeypatch.setattr(
        repo_cli,
        "remove_target_repo_capsule",
        lambda **kwargs: {
            "owner": "acme",
            "repo": "method-zoo",
            "branch": "main",
            "folder": "demo_capsule",
            "committed": True,
            "pushed": True,
        },
    )
    monkeypatch.setattr(repo_cli, "_confirm", lambda question, default=False: True)

    rc = repo_cli.main(["edit", "acme/method-zoo", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "edit-remove-remote"
    assert payload["target"]["remote_remove"]["folder"] == "demo_capsule"

    rc = repo_cli.main(["list", "--json"])
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["targets"][0]["capsules"] == []


def test_repo_cli_list_without_capsule_shows_all_configured_targets(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_repo_list"
    workspace.mkdir()
    caps_a = capsule_create.create_capsule_scaffold(capsule_name="caps_a", base_dir=workspace, register=False)
    caps_b = capsule_create.create_capsule_scaffold(capsule_name="caps_b", base_dir=workspace, register=False)
    publish_state = tmp_path / "publish_targets_list.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))

    repo_cli.save_configured_target(
        caps_a,
        {"name": "github", "kind": "github", "owner": "acme", "repo": "caps-a", "branch": "main", "path": "caps_a", "visibility": "private"},
        make_default=True,
    )
    repo_cli.save_configured_target(
        caps_b,
        {"name": "github", "kind": "github", "owner": "acme", "repo": "caps-b", "branch": "main", "path": "caps_b", "visibility": "private"},
        make_default=True,
    )

    rc = repo_cli.main(["list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["repo"] for item in payload["targets"]] == ["caps-a", "caps-b"]
    assert [item["capsules"] for item in payload["targets"]] == [["caps_a"], ["caps_b"]]


def test_repo_cli_list_groups_multiple_capsules_under_same_repo(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_repo_grouped"
    workspace.mkdir()
    caps_a = capsule_create.create_capsule_scaffold(capsule_name="caps_a", base_dir=workspace, register=False)
    caps_b = capsule_create.create_capsule_scaffold(capsule_name="caps_b", base_dir=workspace, register=False)
    publish_state = tmp_path / "publish_targets_grouped.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))

    repo_cli.save_configured_target(
        caps_a,
        {"name": "github", "kind": "github", "owner": "acme", "repo": "shared", "branch": "main", "path": "caps_a", "visibility": "private"},
        make_default=True,
    )
    repo_cli.save_configured_target(
        caps_b,
        {"name": "github", "kind": "github", "owner": "acme", "repo": "shared", "branch": "main", "path": "caps_b", "visibility": "private"},
        make_default=True,
    )

    rc = repo_cli.main(["list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["targets"]) == 1
    assert payload["targets"][0]["repo"] == "shared"
    assert payload["targets"][0]["capsules"] == ["caps_a", "caps_b"]


def test_repo_cli_detach_accepts_owner_repo_globally(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_repo_global"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(capsule_name="caps_g", base_dir=workspace, register=False)
    publish_state = tmp_path / "publish_targets_global.json"
    monkeypatch.setenv("LELABO_PUBLISH_STATE", str(publish_state))

    repo_cli.save_configured_target(
        capsule_root,
        {"name": "github", "kind": "github", "owner": "acme", "repo": "caps-g", "branch": "main", "path": "caps_g", "visibility": "private"},
        make_default=False,
    )
    monkeypatch.setattr(repo_cli, "_confirm", lambda question, default=False: True)

    rc = repo_cli.main(["detach", "acme/caps-g", str(capsule_root), "--json"])
    assert rc == 0
    detach_payload = json.loads(capsys.readouterr().out)
    assert detach_payload["target"]["name"] == "github"


def test_repo_cli_use_and_remove_are_unknown(tmp_path) -> None:
    with pytest.raises(SystemExit) as use_exc:
        repo_cli.main(["use"])
    assert "Unknown targets subcommand: use" in str(use_exc.value)

    with pytest.raises(SystemExit) as remove_exc:
        repo_cli.main(["remove"])
    assert "Unknown targets subcommand: remove" in str(remove_exc.value)
