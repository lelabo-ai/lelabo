from __future__ import annotations

from pathlib import Path

from conftest import REPO_ROOT, load_module_from_path


def test_default_data_dir_points_to_repo_cache() -> None:
    mod = load_module_from_path(
        "dataset_paths_default",
        REPO_ROOT / "src" / "lab" / "supervised" / "datasets" / "paths.py",
    )
    expected = REPO_ROOT / ".cache" / "data"
    assert mod._default_data_dir() == expected


def test_env_override_for_data_dir(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "lelabo-data"
    monkeypatch.setenv("LELABO_DATA_DIR", str(custom))
    mod = load_module_from_path(
        "dataset_paths_env_override",
        REPO_ROOT / "src" / "lab" / "supervised" / "datasets" / "paths.py",
    )

    assert mod.DATA_DIR == custom.resolve()
    ds = mod.dataset_dir("mnist")
    assert ds == custom.resolve() / "mnist"
    assert ds.exists()
