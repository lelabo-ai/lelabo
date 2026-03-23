from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
capsule_cli = importlib.import_module("lelabo.cli.commands.capsule")
capsule_create = importlib.import_module("lelabo.capsule.create")
capsule_pack = importlib.import_module("lelabo.capsule.pack")
capsule_registry = importlib.import_module("lelabo.capsule.registry")


def test_capsule_cli_install_list_show_remove(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.1}), encoding="utf-8")

    bundle = capsule_pack.pack_capsule(source=run_dir, out_path=tmp_path / "cli_capsule.tar.gz", capsule_id="cli_cap")
    assert bundle.exists()

    caps_dir = tmp_path / "caps"
    rc = capsule_cli.main(["install", str(bundle), "--alias", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Success: Capsule install complete." in out
    assert "Capsule install" in out
    assert "capsule_id: cli_cap" in out

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Capsules" in out
    assert "cli_cap" in out
    assert "status: store" in out
    assert "cli_alias" in out
    assert "kind:" not in out
    assert "| path: " in out

    rc = capsule_cli.main(["show", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Visible capsule" in out
    assert "capsule_id: cli_cap" in out
    assert "status: store" in out
    assert "path:" in out

    installed_dir = caps_dir / "cli_cap"
    assert installed_dir.exists()

    rc = capsule_cli.main(["remove", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Removed capsule" in out
    assert "capsule_id: cli_cap" in out
    assert "deleted_files: True" in out
    assert not installed_dir.exists()

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    assert "Info: No capsules found in the current workspace or store." in capsys.readouterr().out


def test_capsule_attach_accepts_external_capsule_directory(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="dir_attach_capsule",
        base_dir=external_root,
        register=False,
    )
    monkeypatch.chdir(workspace)
    caps_dir = tmp_path / "caps_dir_attach"
    rc = capsule_cli.main(["attach", str(capsule_root), "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.capsule/v1"
    assert payload["command"] == "attach"
    assert payload["capsule"]["capsule_id"] == "dir_attach_capsule"
    assert payload["result"]["attach_action"] == "attached"
    assert payload["result"]["moved"] is False
    row = capsule_registry.get_capsule("dir_attach_capsule", caps_dir)
    assert row is not None
    assert row["path"] == str(capsule_root.resolve())


def test_capsule_attach_rejects_capsule_in_current_workspace(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_local_install"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="local_attach_capsule",
        base_dir=workspace,
        register=False,
    )
    monkeypatch.chdir(workspace)

    caps_dir = tmp_path / "caps_local_attach"
    with pytest.raises(SystemExit) as exc:
        capsule_cli.main(["attach", str(capsule_root), "--capsules-dir", str(caps_dir), "--json"])
    assert "already in the current workspace" in str(exc.value)
    assert "`attach` is only for external capsules" in str(exc.value)


def test_capsule_attach_rejects_capsule_already_in_store(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace_store_attach"
    workspace.mkdir()
    external_root = tmp_path / "external_store_attach"
    external_root.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="stored_attach_capsule",
        base_dir=external_root,
        register=False,
    )
    caps_dir = tmp_path / "caps_store_attach"
    rc = capsule_cli.main(["stash", str(capsule_root), "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    monkeypatch.chdir(workspace)

    stored_root = caps_dir / "stored_attach_capsule"
    with pytest.raises(SystemExit) as exc:
        capsule_cli.main(["attach", str(stored_root), "--capsules-dir", str(caps_dir)])
    assert "already lives in the local store" in str(exc.value)


def test_capsule_list_unifies_workspace_store_and_external_attached_capsules(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_list"
    workspace.mkdir()
    workspace_capsule = capsule_create.create_capsule_scaffold(
        capsule_name="workspace_capsule",
        base_dir=workspace,
        register=False,
    )
    store_source = capsule_create.create_capsule_scaffold(
        capsule_name="store_capsule",
        base_dir=tmp_path / "store_source",
        register=False,
    )
    external_capsule = capsule_create.create_capsule_scaffold(
        capsule_name="external_capsule",
        base_dir=tmp_path / "external_source",
        register=False,
    )
    caps_dir = tmp_path / "caps_unified"
    rc = capsule_cli.main(["stash", str(store_source), "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    capsys.readouterr()
    capsule_registry.add_capsule_entry(
        capsule_id="external_capsule",
        capsule_path=external_capsule,
        manifest={
            "kind": "config_only",
            "created_at": "2026-02-20T00:00:00Z",
            "source": {"path": str(external_capsule)},
        },
        alias="external_alias",
        capsules_dir=caps_dir,
    )
    monkeypatch.chdir(workspace)

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.capsules/v2"
    assert [row["capsule_id"] for row in payload["capsules"]] == [
        "external_capsule",
        "workspace_capsule",
        "store_capsule",
    ]
    assert [row["status"] for row in payload["capsules"]] == ["workspace", "workspace", "store"]
    external_row = payload["capsules"][0]
    assert external_row["aliases"] == ["external_alias"]
    assert external_row["path"] == str(external_capsule.resolve())
    store_row = payload["capsules"][-1]
    assert store_row["path"] == str((caps_dir / "store_capsule").resolve())


def test_capsule_list_shows_store_workspace_conflict_as_two_entries(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_conflict"
    workspace.mkdir()
    workspace_capsule = capsule_create.create_capsule_scaffold(
        capsule_name="dup_capsule",
        base_dir=workspace,
        register=False,
    )
    caps_dir = tmp_path / "caps_conflict_list"
    legacy_store_capsule = capsule_create.create_capsule_scaffold(
        capsule_name="dup_capsule",
        base_dir=caps_dir,
        register=False,
    )
    capsule_registry.add_capsule_entry(
        capsule_id="dup_capsule",
        capsule_path=legacy_store_capsule,
        manifest={
            "kind": "config_only",
            "created_at": "2026-02-20T00:00:00Z",
            "source": {"path": str(legacy_store_capsule)},
        },
        capsules_dir=caps_dir,
    )
    monkeypatch.chdir(workspace)

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dup_rows = [row for row in payload["capsules"] if row["capsule_id"] == "dup_capsule"]
    assert len(dup_rows) == 2
    assert {row["status"] for row in dup_rows} == {"workspace", "store"}
    assert {row["path"] for row in dup_rows} == {
        str(workspace_capsule.resolve()),
        str((caps_dir / "dup_capsule").resolve()),
    }


def test_capsule_show_accepts_visible_workspace_capsule(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace_show_visible"
    workspace.mkdir()
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="demo_capsule",
        base_dir=workspace,
        register=False,
    )
    monkeypatch.chdir(workspace)

    rc = capsule_cli.main(["show", "demo_capsule", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "show"
    assert payload["capsule"]["capsule_id"] == "demo_capsule"
    assert payload["capsule"]["status"] == "workspace"
    assert payload["capsule"]["path"] == str(capsule_root.resolve())


def test_capsule_install_rejects_local_directory_and_points_to_attach(tmp_path) -> None:
    capsule_root = capsule_create.create_capsule_scaffold(
        capsule_name="local_install_reject",
        base_dir=tmp_path,
        register=False,
    )
    try:
        capsule_cli.main(["install", str(capsule_root)])
    except SystemExit as exc:
        msg = str(exc)
        assert "local capsule directory" in msg
        assert "lelabo capsule attach" in msg
    else:
        raise AssertionError("Expected SystemExit when using install with local capsule directory.")


def test_capsule_install_github_noops_when_same_capsule_already_in_workspace(tmp_path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    local_capsule = capsule_create.create_capsule_scaffold(
        capsule_name="same_workspace_capsule",
        base_dir=workspace,
        register=False,
    )
    monkeypatch.chdir(workspace)

    def _fake_clone(*, repo_url, destination, ref):
        import shutil

        shutil.copytree(workspace, destination)
        return {
            "repo_url": "https://github.com/acme/same_workspace_capsule.git",
            "owner": "acme",
            "repo": "same_workspace_capsule",
            "requested_ref": ref,
            "resolved_ref": "abc123",
        }

    monkeypatch.setattr(capsule_cli, "clone_github_repo", _fake_clone)

    caps_dir = tmp_path / "caps_store"
    rc = capsule_cli.main(
        [
            "install",
            "https://github.com/acme/same_workspace_capsule",
            "--capsule",
            "same_workspace_capsule",
            "--capsules-dir",
            str(caps_dir),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["install_action"] == "already_present_workspace"
    assert payload["capsule"]["path"] == str(local_capsule.resolve())
    assert capsule_registry.list_capsules(caps_dir) == []


def test_capsule_cli_remove_keep_files(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run_keep"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.2}), encoding="utf-8")

    bundle = capsule_pack.pack_capsule(source=run_dir, out_path=tmp_path / "cli_capsule_keep.tar.gz", capsule_id="cli_keep")

    caps_dir = tmp_path / "caps_keep"
    rc = capsule_cli.main(["install", str(bundle), "--alias", "keep_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    capsys.readouterr()

    installed_dir = caps_dir / "cli_keep"
    assert installed_dir.exists()

    rc = capsule_cli.main(["remove", "keep_alias", "--keep-files", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    assert installed_dir.exists()


def test_capsule_cli_json_outputs_use_versioned_public_schema(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run_json"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.3}), encoding="utf-8")

    bundle = capsule_pack.pack_capsule(source=run_dir, out_path=tmp_path / "cli_capsule_json.tar.gz", capsule_id="cli_json")
    capsys.readouterr()

    caps_dir = tmp_path / "caps_json"

    rc = capsule_cli.main(
        ["install", str(bundle), "--alias", "json_alias", "--capsules-dir", str(caps_dir), "--json"]
    )
    assert rc == 0
    install_payload = json.loads(capsys.readouterr().out)
    assert install_payload["schema_version"] == "lelabo.cli.capsule/v1"
    assert install_payload["command"] == "install"
    assert install_payload["capsule"]["capsule_id"] == "cli_json"
    assert install_payload["capsule"]["kind"] == "single_run"
    assert install_payload["result"]["source_bundle"] == str(bundle.resolve())

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    list_payload = json.loads(capsys.readouterr().out)
    assert list_payload["schema_version"] == "lelabo.cli.capsules/v2"
    assert [row["capsule_id"] for row in list_payload["capsules"]] == ["cli_json"]
    assert list_payload["capsules"][0]["status"] == "store"

    rc = capsule_cli.main(["show", "json_alias", "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    show_payload = json.loads(capsys.readouterr().out)
    assert show_payload["schema_version"] == "lelabo.cli.capsule/v1"
    assert show_payload["command"] == "show"
    assert show_payload["capsule"]["capsule_id"] == "cli_json"
    assert "result" not in show_payload

    rc = capsule_cli.main(["remove", "json_alias", "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    remove_payload = json.loads(capsys.readouterr().out)
    assert remove_payload["schema_version"] == "lelabo.cli.capsule/v1"
    assert remove_payload["command"] == "remove"
    assert remove_payload["capsule"]["capsule_id"] == "cli_json"
    assert remove_payload["result"]["deleted_files"] is True


def test_capsule_install_same_source_is_noop(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run_same"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.31}), encoding="utf-8")
    bundle = capsule_pack.pack_capsule(source=run_dir, out_path=tmp_path / "same_capsule.tar.gz", capsule_id="same_capsule")
    capsys.readouterr()

    caps_dir = tmp_path / "caps_same"
    rc = capsule_cli.main(["install", str(bundle), "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    first = json.loads(capsys.readouterr().out)
    assert first["result"]["install_action"] == "installed"

    rc = capsule_cli.main(["install", str(bundle), "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    second = json.loads(capsys.readouterr().out)
    assert second["result"]["install_action"] == "unchanged"
    assert second["result"]["replaced_existing"] is False


def test_capsule_install_conflict_requires_replace_or_rename(tmp_path, capsys) -> None:
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    (run_a / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_a / "summary.json").write_text(json.dumps({"acc": 0.11}), encoding="utf-8")
    bundle_a = capsule_pack.pack_capsule(source=run_a, out_path=tmp_path / "a.tar.gz", capsule_id="dup_capsule")
    capsys.readouterr()

    run_b = tmp_path / "run_b"
    run_b.mkdir()
    (run_b / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_b / "summary.json").write_text(json.dumps({"acc": 0.99}), encoding="utf-8")
    bundle_b = capsule_pack.pack_capsule(source=run_b, out_path=tmp_path / "b.tar.gz", capsule_id="dup_capsule")
    capsys.readouterr()

    caps_dir = tmp_path / "caps_conflict"
    rc = capsule_cli.main(["install", str(bundle_a), "--capsules-dir", str(caps_dir)])
    assert rc == 0
    capsys.readouterr()

    try:
        capsule_cli.main(["install", str(bundle_b), "--capsules-dir", str(caps_dir)])
    except SystemExit as exc:
        msg = str(exc)
        assert "--force-replace" in msg
        assert "--rename-to" in msg
    else:
        raise AssertionError("Expected SystemExit for conflicting capsule install.")


def test_capsule_install_rename_to_allows_side_by_side(tmp_path, capsys) -> None:
    run_a = tmp_path / "run_a2"
    run_a.mkdir()
    (run_a / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_a / "summary.json").write_text(json.dumps({"acc": 0.21}), encoding="utf-8")
    bundle_a = capsule_pack.pack_capsule(source=run_a, out_path=tmp_path / "a2.tar.gz", capsule_id="rename_base")
    capsys.readouterr()

    run_b = tmp_path / "run_b2"
    run_b.mkdir()
    (run_b / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_b / "summary.json").write_text(json.dumps({"acc": 0.22}), encoding="utf-8")
    bundle_b = capsule_pack.pack_capsule(source=run_b, out_path=tmp_path / "b2.tar.gz", capsule_id="rename_base")
    capsys.readouterr()

    caps_dir = tmp_path / "caps_rename"
    rc = capsule_cli.main(["install", str(bundle_a), "--capsules-dir", str(caps_dir)])
    assert rc == 0
    capsys.readouterr()

    rc = capsule_cli.main(
        ["install", str(bundle_b), "--capsules-dir", str(caps_dir), "--rename-to", "rename_variant", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "rename_variant"
    assert payload["result"]["install_action"] == "installed"


def test_capsule_cli_remove_rejects_external_paths(tmp_path) -> None:
    caps_dir = tmp_path / "capsules"
    external = tmp_path / "external_capsule"
    external.mkdir()
    (external / "manifest.json").write_text("{}", encoding="utf-8")

    capsule_registry.add_capsule_entry(
        capsule_id="external_cap",
        capsule_path=external,
        manifest={"kind": "config_only", "created_at": "2026-02-20T00:00:00Z", "source": {"path": str(external)}},
        alias="external_alias",
        capsules_dir=caps_dir,
    )

    try:
        capsule_cli.main(["remove", "external_alias", "--capsules-dir", str(caps_dir)])
    except SystemExit as exc:
        assert "Refusing to delete capsule path outside capsules store" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for unsafe external capsule deletion.")


def test_capsule_cli_remove_allows_external_paths_with_rf(tmp_path) -> None:
    caps_dir = tmp_path / "capsules"
    external = tmp_path / "external_capsule_force"
    external.mkdir()
    (external / "manifest.json").write_text("{}", encoding="utf-8")

    capsule_registry.add_capsule_entry(
        capsule_id="external_cap_force",
        capsule_path=external,
        manifest={"kind": "config_only", "created_at": "2026-02-20T00:00:00Z", "source": {"path": str(external)}},
        alias="external_alias_force",
        capsules_dir=caps_dir,
    )

    rc = capsule_cli.main(["remove", "external_alias_force", "--capsules-dir", str(caps_dir), "-r", "-f"])
    assert rc == 0
    assert not external.exists()
    assert capsule_registry.get_capsule("external_alias_force", caps_dir) is None


def test_capsule_cli_remove_rejects_half_rf_flag(tmp_path) -> None:
    caps_dir = tmp_path / "capsules"
    external = tmp_path / "external_capsule_half_rf"
    external.mkdir()
    (external / "manifest.json").write_text("{}", encoding="utf-8")

    capsule_registry.add_capsule_entry(
        capsule_id="external_cap_half_rf",
        capsule_path=external,
        manifest={"kind": "config_only", "created_at": "2026-02-20T00:00:00Z", "source": {"path": str(external)}},
        alias="external_alias_half_rf",
        capsules_dir=caps_dir,
    )

    try:
        capsule_cli.main(["remove", "external_alias_half_rf", "--capsules-dir", str(caps_dir), "-f"])
    except SystemExit as exc:
        assert "Use '-rf' together" in str(exc)
    else:
        raise AssertionError("Expected SystemExit when only one of -r/-f is provided.")

    assert external.exists()
    assert capsule_registry.get_capsule("external_alias_half_rf", caps_dir) is not None


def test_capsule_cli_checkout_rejects_external_paths(tmp_path) -> None:
    caps_dir = tmp_path / "capsules"
    external = tmp_path / "external_capsule"
    external.mkdir()
    (external / "manifest.json").write_text("{}", encoding="utf-8")

    capsule_registry.add_capsule_entry(
        capsule_id="external_checkout_cap",
        capsule_path=external,
        manifest={"kind": "config_only", "created_at": "2026-02-20T00:00:00Z", "source": {"path": str(external)}},
        alias="external_checkout_alias",
        capsules_dir=caps_dir,
    )

    try:
        capsule_cli.main(["checkout", "external_checkout_alias", "--capsules-dir", str(caps_dir)])
    except SystemExit as exc:
        assert "Refusing to checkout capsule files outside cache" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for unsafe external capsule checkout.")


def test_capsule_cli_stash_and_checkout(tmp_path, capsys) -> None:
    source = capsule_create.create_capsule_scaffold(
        capsule_name="store_demo_capsule",
        base_dir=tmp_path,
        register=False,
    )

    caps_dir = tmp_path / "caps_store"
    rc = capsule_cli.main(["stash", str(source), "--alias", "stored_alias", "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0

    store_payload = json.loads(capsys.readouterr().out)
    assert store_payload["schema_version"] == "lelabo.cli.capsule/v1"
    assert store_payload["command"] == "stash"
    assert store_payload["capsule"]["capsule_id"] == "store_demo_capsule"
    assert store_payload["capsule"]["kind"] == "config_only"
    assert store_payload["result"]["moved"] is True
    stored_path = caps_dir / "store_demo_capsule"
    assert stored_path.exists()
    assert (stored_path / "capsule.toml").exists()
    assert not source.exists()

    restore_root = tmp_path / "restore_workspace"
    rc = capsule_cli.main(
        [
            "checkout",
            "stored_alias",
            str(restore_root),
            "--capsules-dir",
            str(caps_dir),
            "--json",
        ]
    )
    assert rc == 0

    restore_payload = json.loads(capsys.readouterr().out)
    restored_path = restore_root / "store_demo_capsule"
    assert restore_payload["schema_version"] == "lelabo.cli.capsule/v1"
    assert restore_payload["command"] == "checkout"
    assert restore_payload["capsule"]["capsule_id"] == "store_demo_capsule"
    assert restore_payload["capsule"]["kind"] == "config_only"
    assert restore_payload["capsule"]["path"] == str(restored_path.resolve())
    assert restore_payload["result"]["checked_out_path"] == str(restored_path.resolve())
    assert restore_payload["result"]["moved"] is True
    assert restore_payload["result"]["removed_from_cache"] is True
    assert (restored_path / "capsule.toml").exists()
    assert not stored_path.exists()
    assert capsule_registry.get_capsule("stored_alias", caps_dir) is None


def test_capsule_cli_stash_uses_active_capsule_when_source_is_omitted(tmp_path, monkeypatch, capsys) -> None:
    source = capsule_create.create_capsule_scaffold(
        capsule_name="active_capsule",
        base_dir=tmp_path,
        register=False,
    )
    monkeypatch.chdir(source / "models")

    caps_dir = tmp_path / "caps_active"
    rc = capsule_cli.main(["stash", "--alias", "active_alias", "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

    row = capsule_registry.get_capsule("active_alias", caps_dir)
    assert row is not None
    assert row["capsule_id"] == "active_capsule"
    assert payload["capsule"]["capsule_id"] == "active_capsule"
    assert not source.exists()


def test_capsule_cli_stash_accepts_manifest_without_capsule_toml(tmp_path, capsys) -> None:
    source = capsule_create.create_capsule_scaffold(
        capsule_name="manifest_only_capsule",
        base_dir=tmp_path,
        register=False,
    )
    (source / "capsule.toml").unlink()

    caps_dir = tmp_path / "caps_manifest_only"
    rc = capsule_cli.main(
        [
            "stash",
            str(source),
            "--alias",
            "manifest_only_alias",
            "--capsules-dir",
            str(caps_dir),
            "--json",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "manifest_only_capsule"
    assert payload["result"]["moved"] is True
    assert (caps_dir / "manifest_only_capsule" / "manifest.json").exists()
    assert not source.exists()


def test_capsule_cli_stash_from_parent_directory_uses_named_subfolder(tmp_path, capsys) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = capsule_create.create_capsule_scaffold(
        capsule_name="test",
        base_dir=root,
        register=False,
    )

    caps_dir = tmp_path / "caps_parent_lookup"
    rc = capsule_cli.main(
        [
            "stash",
            str(root),
            "--alias",
            "test",
            "--capsules-dir",
            str(caps_dir),
            "--json",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "test"
    assert payload["result"]["stored_from"] == str(source.resolve())
    assert payload["result"]["moved"] is True
    assert (caps_dir / "test" / "capsule.toml").exists()
    assert not source.exists()


def test_capsule_cli_stash_all_moves_direct_child_capsules_only(tmp_path, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cap_a = capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=workspace, register=False)
    cap_b = capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=workspace, register=False)
    (workspace / "not_a_capsule").mkdir()

    caps_dir = tmp_path / "caps_all"
    rc = capsule_cli.main(["stash", "--all", str(workspace), "--capsules-dir", str(caps_dir), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.capsules/v1"
    assert payload["command"] == "stash"
    assert payload["count"] == 2
    assert sorted(item["capsule_id"] for item in payload["capsules"]) == ["cap_a", "cap_b"]
    assert all(item["result"]["moved"] is True for item in payload["capsules"])
    assert not cap_a.exists()
    assert not cap_b.exists()
    assert (caps_dir / "cap_a").exists()
    assert (caps_dir / "cap_b").exists()


def test_capsule_cli_stash_all_rejects_alias(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=workspace, register=False)

    try:
        capsule_cli.main(["stash", "--all", str(workspace), "--alias", "bad_alias"])
    except SystemExit as exc:
        assert "--alias" in str(exc)
    else:
        raise AssertionError("Expected SystemExit when using --alias with stash --all.")


def test_capsule_cli_stash_all_rejects_running_from_active_capsule(tmp_path, monkeypatch) -> None:
    source = capsule_create.create_capsule_scaffold(capsule_name="active_capsule", base_dir=tmp_path, register=False)
    monkeypatch.chdir(source)
    try:
        capsule_cli.main(["stash", "--all"])
    except SystemExit as exc:
        assert "expects a container directory" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for stash --all inside an active capsule.")


def test_capsule_cli_install_github_store_only(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "github_repo"
    repo_root.mkdir()
    source = capsule_create.create_capsule_scaffold(
        capsule_name="github_capsule",
        base_dir=repo_root,
        register=False,
    )
    caps_dir = tmp_path / "caps_github"

    def _fake_clone(*, repo_url, destination, ref):
        import shutil

        shutil.copytree(repo_root, destination)
        return {
            "repo_url": "https://github.com/acme/github_capsule.git",
            "owner": "acme",
            "repo": "github_capsule",
            "requested_ref": ref,
            "resolved_ref": "abc123",
        }

    monkeypatch.setattr(capsule_cli, "clone_github_repo", _fake_clone)

    rc = capsule_cli.main(
        [
            "install",
            "https://github.com/acme/github_capsule",
            "--capsule",
            "github_capsule",
            "--ref",
            "main",
            "--capsules-dir",
            str(caps_dir),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.capsule/v1"
    assert payload["command"] == "install"
    assert payload["result"]["source_kind"] == "github_repo"
    assert payload["result"]["source_url"] == "https://github.com/acme/github_capsule.git"
    assert payload["result"]["source_ref"] == "abc123"
    assert payload["result"]["checked_out"] is False
    assert (caps_dir / "github_capsule").exists()


def test_capsule_cli_install_rejects_non_github_remote_source() -> None:
    try:
        capsule_cli.main(["install", "https://gitlab.com/acme/demo"])
    except SystemExit as exc:
        assert "GitHub repo URLs only" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for unsupported non-GitHub remote install source.")


def test_capsule_cli_install_github_repo_without_manifest_still_works(tmp_path, monkeypatch, capsys) -> None:
    source = capsule_create.create_capsule_scaffold(
        capsule_name="plain_repo_capsule",
        base_dir=tmp_path,
        register=False,
    )

    def _fake_clone(*, repo_url, destination, ref):
        import shutil

        shutil.copytree(source, destination)
        return {
            "repo_url": "https://github.com/acme/plain_repo_capsule.git",
            "owner": "acme",
            "repo": "plain_repo_capsule",
            "requested_ref": ref,
            "resolved_ref": "abc123",
        }

    monkeypatch.setattr(capsule_cli, "clone_github_repo", _fake_clone)

    rc = capsule_cli.main(["install", "https://github.com/acme/plain_repo_capsule", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule"]["capsule_id"] == "plain_repo_capsule"
    assert payload["result"]["source_kind"] == "github_repo"


def test_capsule_cli_install_github_repo_all(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "multi_repo"
    repo_root.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=repo_root, register=False)
    capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=repo_root, register=False)

    def _fake_clone(*, repo_url, destination, ref):
        import shutil

        shutil.copytree(repo_root, destination)
        return {
            "repo_url": "https://github.com/acme/multi_repo.git",
            "owner": "acme",
            "repo": "multi_repo",
            "requested_ref": ref,
            "resolved_ref": "abc123",
        }

    monkeypatch.setattr(capsule_cli, "clone_github_repo", _fake_clone)

    caps_dir = tmp_path / "caps_multi"
    rc = capsule_cli.main(
        ["install", "https://github.com/acme/multi_repo", "--all", "--capsules-dir", str(caps_dir), "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.capsules/v1"
    assert payload["command"] == "install"
    assert sorted(item["capsule_id"] for item in payload["capsules"]) == ["cap_a", "cap_b"]


def test_capsule_cli_install_github_repo_repeatable_capsule_flags(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "multi_repo_repeatable"
    repo_root.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=repo_root, register=False)
    capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=repo_root, register=False)

    def _fake_clone(*, repo_url, destination, ref):
        import shutil

        shutil.copytree(repo_root, destination)
        return {
            "repo_url": "https://github.com/acme/multi_repo_repeatable.git",
            "owner": "acme",
            "repo": "multi_repo_repeatable",
            "requested_ref": ref,
            "resolved_ref": "abc123",
        }

    monkeypatch.setattr(capsule_cli, "clone_github_repo", _fake_clone)

    caps_dir = tmp_path / "caps_multi_repeatable"
    rc = capsule_cli.main(
        [
            "install",
            "https://github.com/acme/multi_repo_repeatable",
            "--capsule",
            "cap_b",
            "--capsule",
            "cap_a",
            "--capsules-dir",
            str(caps_dir),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.capsules/v1"
    assert sorted(item["capsule_id"] for item in payload["capsules"]) == ["cap_a", "cap_b"]


def test_capsule_cli_install_github_repo_non_tty_requires_explicit_selection(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "multi_repo_non_tty"
    repo_root.mkdir()
    capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=repo_root, register=False)
    capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=repo_root, register=False)

    def _fake_clone(*, repo_url, destination, ref):
        import shutil

        shutil.copytree(repo_root, destination)
        return {
            "repo_url": "https://github.com/acme/multi_repo_non_tty.git",
            "owner": "acme",
            "repo": "multi_repo_non_tty",
            "requested_ref": ref,
            "resolved_ref": "abc123",
        }

    monkeypatch.setattr(capsule_cli, "clone_github_repo", _fake_clone)
    monkeypatch.setattr(capsule_cli, "_is_interactive_tty", lambda: False)

    try:
        capsule_cli.main(["install", "https://github.com/acme/multi_repo_non_tty"])
    except SystemExit as exc:
        msg = str(exc)
        assert "--capsule <id>" in msg
        assert "(repeatable)" in msg
        assert "--all" in msg
    else:
        raise AssertionError("Expected SystemExit when multi-capsule install runs non-interactively.")


def test_capsule_cli_install_github_repo_interactive_picker_is_used(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "multi_repo_picker"
    repo_root.mkdir()
    cap_a = capsule_create.create_capsule_scaffold(capsule_name="cap_a", base_dir=repo_root, register=False)
    cap_b = capsule_create.create_capsule_scaffold(capsule_name="cap_b", base_dir=repo_root, register=False)

    def _fake_clone(*, repo_url, destination, ref):
        import shutil

        shutil.copytree(repo_root, destination)
        return {
            "repo_url": "https://github.com/acme/multi_repo_picker.git",
            "owner": "acme",
            "repo": "multi_repo_picker",
            "requested_ref": ref,
            "resolved_ref": "abc123",
        }

    monkeypatch.setattr(capsule_cli, "clone_github_repo", _fake_clone)
    monkeypatch.setattr(capsule_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr(capsule_cli, "pick_many_with_checkboxes", lambda **kwargs: ["cap_b"])

    caps_dir = tmp_path / "caps_picker"
    rc = capsule_cli.main(
        ["install", "https://github.com/acme/multi_repo_picker", "--capsules-dir", str(caps_dir), "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "lelabo.cli.capsule/v1"
    assert payload["capsule"]["capsule_id"] == "cap_b"


def test_capsule_cli_share_is_unknown() -> None:
    with pytest.raises(SystemExit) as exc:
        capsule_cli.main(["share"])

    assert "Unknown capsule subcommand: share" in str(exc.value)


def test_capsule_cli_uses_store_dir_from_settings_when_capsules_dir_not_provided(
    tmp_path, monkeypatch, capsys
) -> None:
    run_dir = tmp_path / "run_store_cfg"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.4}), encoding="utf-8")
    bundle = capsule_pack.pack_capsule(source=run_dir, out_path=tmp_path / "store_cfg_capsule.tar.gz", capsule_id="cfg_store_cap")
    capsys.readouterr()

    configured_store = tmp_path / "configured_store"
    monkeypatch.setattr(
        capsule_cli,
        "_effective_settings",
        lambda: {
            "github": {
                "owner": "",
                "default_visibility": "private",
                "default_branch": "main",
                "create_repo_if_missing": True,
            },
            "capsules": {
                "store_dir": str(configured_store),
                "default_checkout_dir": ".",
                "install_checkout": False,
            },
        },
    )

    rc = capsule_cli.main(["install", str(bundle), "--alias", "cfg_store_alias"])
    assert rc == 0
    capsys.readouterr()
    row = capsule_registry.get_capsule("cfg_store_alias", configured_store)
    assert row is not None
    assert row["capsule_id"] == "cfg_store_cap"
