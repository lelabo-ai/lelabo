from __future__ import annotations

import sys
import types
from pathlib import Path


def ensure_src_on_path() -> Path:
    """Ensure `src/` is importable and return repo root."""
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    src_str = str(src_root)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return repo_root


def _stub_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        mod = sys.modules[name]
        if hasattr(mod, "__path__"):
            return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, pkg)


def ensure_lab_namespace() -> Path:
    repo_root = ensure_src_on_path()
    src = repo_root / "src" / "lab"

    _stub_namespace_package("lab", src)
    _stub_namespace_package("lab.algorithms", src / "algorithms")
    _stub_namespace_package("lab.algorithms.update_rules", src / "algorithms" / "update_rules")
    _stub_namespace_package("lab.core", src / "core")
    _stub_namespace_package("lab.core.utils", src / "core" / "utils")
    _stub_namespace_package("lab.models", src / "models")
    _stub_namespace_package("lab.datasets", src / "datasets")
    return repo_root
