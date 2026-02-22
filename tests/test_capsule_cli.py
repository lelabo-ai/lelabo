from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
capsule_cli = importlib.import_module("lab.cli.commands.capsule")
capsule_create = importlib.import_module("lab.capsule.create")
capsule_registry = importlib.import_module("lab.capsule.registry")


def test_capsule_cli_pack_install_list_show(tmp_path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lab.main"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.1}), encoding="utf-8")

    bundle = tmp_path / "cli_capsule.tar.gz"
    rc = capsule_cli.main(["pack", "--from", str(run_dir), "--out", str(bundle), "--id", "cli_cap"])
    assert rc == 0
    assert bundle.exists()

    caps_dir = tmp_path / "caps"
    rc = capsule_cli.main(["install", str(bundle), "--name", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cli_cap" in out

    rc = capsule_cli.main(["show", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out2 = capsys.readouterr().out
    assert "cli_cap" in out2

    installed_dir = caps_dir / "cli_cap"
    assert installed_dir.exists()

    rc = capsule_cli.main(["remove", "cli_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out3 = capsys.readouterr().out
    assert '"capsule_id": "cli_cap"' in out3
    assert '"deleted_files": true' in out3
    assert not installed_dir.exists()

    rc = capsule_cli.main(["list", "--capsules-dir", str(caps_dir)])
    assert rc == 0
    out4 = capsys.readouterr().out
    assert "(no capsules installed)" in out4


def test_capsule_cli_remove_keep_files(tmp_path) -> None:
    run_dir = tmp_path / "run_keep"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"argv": ["python", "-m", "lab.main"]}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"acc": 0.2}), encoding="utf-8")

    bundle = tmp_path / "cli_capsule_keep.tar.gz"
    rc = capsule_cli.main(["pack", "--from", str(run_dir), "--out", str(bundle), "--id", "cli_keep"])
    assert rc == 0

    caps_dir = tmp_path / "caps_keep"
    rc = capsule_cli.main(["install", str(bundle), "--name", "keep_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0

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


def test_capsule_cli_store_and_restore(tmp_path, capsys) -> None:
    source = capsule_create.create_capsule_scaffold(
        capsule_name="store_demo_capsule",
        base_dir=tmp_path,
        register=False,
    )

    caps_dir = tmp_path / "caps_store"
    rc = capsule_cli.main(["store", "-n", "stored_alias", "--from", str(source), "--capsules-dir", str(caps_dir)])
    assert rc == 0

    store_payload = json.loads(capsys.readouterr().out)
    assert store_payload["capsule_id"] == "store_demo_capsule"
    stored_path = caps_dir / "store_demo_capsule"
    assert stored_path.exists()
    assert (stored_path / "capsule.toml").exists()

    restore_root = tmp_path / "restore_workspace"
    rc = capsule_cli.main(
        [
            "restore",
            "stored_alias",
            "--to",
            str(restore_root),
            "--name",
            "restored_local_capsule",
            "--capsules-dir",
            str(caps_dir),
        ]
    )
    assert rc == 0

    restore_payload = json.loads(capsys.readouterr().out)
    restored_path = restore_root / "restored_local_capsule"
    assert restore_payload["capsule_id"] == "store_demo_capsule"
    assert restore_payload["restored_path"] == str(restored_path.resolve())
    assert (restored_path / "capsule.toml").exists()


def test_capsule_cli_store_uses_active_capsule_when_from_is_omitted(tmp_path, monkeypatch) -> None:
    source = capsule_create.create_capsule_scaffold(
        capsule_name="active_capsule",
        base_dir=tmp_path,
        register=False,
    )
    monkeypatch.chdir(source / "models")

    caps_dir = tmp_path / "caps_active"
    rc = capsule_cli.main(["store", "-n", "active_alias", "--capsules-dir", str(caps_dir)])
    assert rc == 0

    row = capsule_registry.get_capsule("active_alias", caps_dir)
    assert row is not None
    assert row["capsule_id"] == "active_capsule"
