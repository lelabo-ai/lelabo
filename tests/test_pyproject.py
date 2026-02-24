from __future__ import annotations

import tomllib

from conftest import REPO_ROOT


def test_pyproject_declares_src_layout_and_cli_scripts() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    package_dir = pyproject["tool"]["setuptools"]["package-dir"]
    assert package_dir[""] == "src"

    scripts = pyproject["project"]["scripts"]
    assert scripts["lelabo"] == "lelabo.cli.main:main"
    assert "lelabo-profiler" not in scripts


def test_pyproject_has_test_optional_dependencies() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    test_deps = pyproject["project"]["optional-dependencies"]["test"]
    assert any(dep.startswith("pytest") for dep in test_deps)
