from __future__ import annotations

import importlib
import json
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
capsule_cli = importlib.import_module("lab.cli.commands.capsule")


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
