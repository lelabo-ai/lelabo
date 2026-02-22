from __future__ import annotations

import importlib
import sys

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))


def test_discover_imports_nested_modules_without_init_side_effects(tmp_path) -> None:
    pkg_root = tmp_path / "demo_registry_pkg"
    nested = pkg_root / "nested"
    nested.mkdir(parents=True)

    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "api.py").write_text(
        "from lab.core.registry import Registry\n\n"
        "REG = Registry('demo', package='demo_registry_pkg')\n\n"
        "def register(name):\n"
        "    return REG.register(name)\n",
        encoding="utf-8",
    )
    (nested / "dataset.py").write_text(
        "from demo_registry_pkg.api import register\n\n"
        "@register('nested_dataset')\n"
        "def build_nested_dataset():\n"
        "    return object()\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        api = importlib.import_module("demo_registry_pkg.api")
        api.REG.discover()
        assert "nested_dataset" in api.REG.names()
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in list(sys.modules):
            if mod_name == "demo_registry_pkg" or mod_name.startswith("demo_registry_pkg."):
                sys.modules.pop(mod_name, None)
