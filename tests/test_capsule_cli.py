from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
capsule_cli = importlib.import_module("lelabo.cli.commands.capsule")
capsule_create = importlib.import_module("lelabo.capsule.create")
capsule_registry = importlib.import_module("lelabo.capsule.registry")


def test_capsule_cli_pack_install_list_show_remove(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.1}), encoding="utf-8")

    bundle = tmp_path / "cli_capsule.tar.gz"
    rc = capsule_cli.main(["pack", "--from", str(run_dir), "--out", str(bundle), "--id", "cli_cap"])
    assert rc == 0
    assert bundle.exists()

    caps_dir = tmp_path / "caps"
    rc = capsule_cli.main(["install", str(bundle), "--alias", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "installed capsule:" in out
    assert "capsule_id: cli_cap" in out

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "capsules:" in out
    assert "cli_cap" in out
    assert "cli_alias" in out

    rc = capsule_cli.main(["show", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "capsule:" in out
    assert "capsule_id: cli_cap" in out

    installed_dir = caps_dir / "cli_cap"
    assert installed_dir.exists()

    rc = capsule_cli.main(["remove", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "removed capsule:" in out
    assert "capsule_id: cli_cap" in out
    assert "deleted_files: True" in out
    assert not installed_dir.exists()

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    assert "(no capsules installed)" in capsys.readouterr().out


def test_capsule_cli_remove_keep_files(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run_keep"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lelabo"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.2}), encoding="utf-8")

    bundle = tmp_path / "cli_capsule_keep.tar.gz"
    rc = capsule_cli.main(["pack", "--from", str(run_dir), "--out", str(bundle), "--id", "cli_keep"])
    assert rc == 0

    caps_dir = tmp_path / "caps_keep"
    rc = capsule_cli.main(["install", str(bundle), "--alias", "keep_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    capsys.readouterr()

    installed_dir = caps_dir / "cli_keep"
    assert installed_dir.exists()

    rc = capsule_cli.main(["remove", "keep_alias", "--keep-files", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    assert installed_dir.exists()


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
    assert store_payload["capsule_id"] == "store_demo_capsule"
    assert store_payload["moved"] is True
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
    assert restore_payload["capsule_id"] == "store_demo_capsule"
    assert restore_payload["checked_out_path"] == str(restored_path.resolve())
    assert restore_payload["moved"] is True
    assert restore_payload["removed_from_cache"] is True
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
    assert payload["capsule_id"] == "active_capsule"
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
    assert payload["capsule_id"] == "manifest_only_capsule"
    assert payload["moved"] is True
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
    assert payload["capsule_id"] == "test"
    assert payload["stored_from"] == str(source.resolve())
    assert payload["moved"] is True
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
    assert payload["count"] == 2
    assert sorted(item["capsule_id"] for item in payload["stashed"]) == ["cap_a", "cap_b"]
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
