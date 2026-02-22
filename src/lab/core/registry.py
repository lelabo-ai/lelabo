from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict

class Registry:
    def __init__(self, name: str, package: str | None = None):
        self.name = name
        self.package = package
        self._items: Dict[str, Callable[..., object]] = {}

    def register(self, name: str):
        key = name.lower()
        def _decorator(obj):
            if key in self._items:
                raise ValueError(f"[{self.name}] '{name}' already registered")
            self._items[key] = obj
            return obj
        return _decorator

    def get(self, name: str):
        key = name.lower()
        if key not in self._items:
            raise ValueError(f"[{self.name}] Unknown '{name}'. Available: {sorted(self._items)}")
        return self._items[key]

    def names(self) -> list[str]:
        return sorted(self._items.keys())

    def discover(self, package: str | None = None):
        pkg_name = package or self.package
        if not pkg_name:
            raise ValueError("discover() needs a package name")
        pkg = importlib.import_module(pkg_name)
        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            return

        for _, full_name, _ in pkgutil.walk_packages(pkg_path, prefix=f"{pkg.__name__}."):
            rel_name = full_name[len(pkg.__name__) + 1 :]
            # Skip private modules/packages anywhere in the relative module path.
            if any(part.startswith("_") for part in rel_name.split(".")):
                continue
            try:
                importlib.import_module(full_name)
            except ImportError:
                # Ignore missing optional deps in plugin-like modules.
                pass
