from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT, load_module_from_path


launch_grid = load_module_from_path(
    "launch_grid_test_module",
    REPO_ROOT / "experiments" / "launchers" / "launch_grid.py",
)


def test_resolve_config_path_supports_legacy_and_new_paths() -> None:
    resolved_new = launch_grid.resolve_config_path("experiments/configs/demo.yaml").resolve()
    resolved_old = launch_grid.resolve_config_path("configs/demo.yaml").resolve()
    resolved_bare = launch_grid.resolve_config_path("demo.yaml").resolve()

    expected = (launch_grid.REPO_ROOT / "experiments" / "configs" / "demo.yaml").resolve()
    assert resolved_new == expected
    assert resolved_old == expected
    assert resolved_bare == expected


def test_with_src_on_pythonpath_sets_required_env() -> None:
    out = launch_grid.with_src_on_pythonpath({})
    src_path = str((launch_grid.REPO_ROOT / "src").resolve())
    py_paths = out["PYTHONPATH"].split(os.pathsep)

    assert py_paths[0] == src_path
    assert out["XDG_CACHE_HOME"] == str((launch_grid.REPO_ROOT / ".cache").resolve())


def test_launcher_dry_run_writes_plan(tmp_path: Path) -> None:
    outdir = tmp_path / "runs"
    cmd = [
        sys.executable,
        str(launch_grid.REPO_ROOT / "experiments" / "launchers" / "launch_grid.py"),
        "--config",
        "experiments/configs/demo.yaml",
        "--outdir",
        str(outdir),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, cwd=launch_grid.REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    plan = outdir / "demo" / "plan.json"
    assert plan.exists()
