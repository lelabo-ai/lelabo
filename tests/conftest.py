from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _isolate_default_installed_capsules(monkeypatch: pytest.MonkeyPatch):
    registry_modules = [
        "lelabo.callbacks.registry",
        "lelabo.initializers.registry",
        "lelabo.losses.registry",
        "lelabo.metrics.registry",
        "lelabo.models.registry",
        "lelabo.optimizers.registry",
        "lelabo.schedulers.registry",
        "lelabo.supervised.datasets.registry",
        "lelabo.update_rules.registry",
    ]

    for module_name in registry_modules:
        mod = importlib.import_module(module_name)
        original = getattr(mod, "load_installed_capsule_plugins", None)
        if not callable(original):
            continue

        def _isolated_load_installed_capsule_plugins(
            *,
            kinds,
            capsules_dir=None,
            _original=original,
        ):
            if capsules_dir is None and not os.getenv("LELABO_CAPSULES_DIR", "").strip():
                return []
            return _original(kinds=kinds, capsules_dir=capsules_dir)

        monkeypatch.setattr(mod, "load_installed_capsule_plugins", _isolated_load_installed_capsule_plugins)
