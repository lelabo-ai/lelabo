from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import warnings
from typing import Callable, Dict


def _strict_plugin_loading() -> bool:
    raw = os.getenv("LELABO_STRICT_PLUGINS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_internal_missing_module(pkg_name: str, missing_name: str) -> bool:
    package_root = str(pkg_name).partition(".")[0]
    token = str(missing_name or "").strip()
    if not token:
        return True
    return token == package_root or token.startswith(f"{package_root}.")


def _warn_or_raise_optional_import(
    *,
    registry_name: str,
    package_name: str,
    module_name: str,
    exc: ImportError,
) -> None:
    if not isinstance(exc, ModuleNotFoundError):
        raise exc
    missing_name = str(getattr(exc, "name", "") or "").strip()
    if _is_internal_missing_module(package_name, missing_name):
        raise exc
    if _strict_plugin_loading():
        raise exc
    warnings.warn(
        f"[{registry_name}] Skipping '{module_name}' due to missing optional dependency "
        f"'{missing_name}': {exc}",
        RuntimeWarning,
        stacklevel=2,
    )


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

    def discover(self, package: str | None = None, *, reload: bool = False):
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
                if reload and full_name in sys.modules:
                    importlib.reload(sys.modules[full_name])
                else:
                    importlib.import_module(full_name)
            except ImportError as exc:
                _warn_or_raise_optional_import(
                    registry_name=self.name,
                    package_name=pkg.__name__,
                    module_name=full_name,
                    exc=exc,
                )

    def snapshot_discovered_items(self, package: str | None = None) -> Dict[str, Callable[..., object]]:
        original_items = dict(self._items)
        preloaded = set(sys.modules.keys())
        owner_modules: set[str] = set()
        for mod_name, mod in list(sys.modules.items()):
            mod_dict = getattr(mod, "__dict__", None)
            if not isinstance(mod_dict, dict):
                continue
            if any(value is self for value in mod_dict.values()):
                owner_modules.add(mod_name)
        try:
            self._items = {}
            self.discover(package=package)

            pkg_name = package or self.package
            if not pkg_name:
                raise ValueError("discover() needs a package name")
            pkg = importlib.import_module(pkg_name)
            pkg_path = getattr(pkg, "__path__", None)
            if pkg_path is not None:
                entries = list(pkgutil.walk_packages(pkg_path, prefix=f"{pkg.__name__}."))
                package_children = {
                    full_name
                    for _, full_name, ispkg in entries
                    if ispkg
                    and any(
                        child_name.startswith(f"{full_name}.")
                        for _, child_name, _ in entries
                    )
                }
                for _, full_name, ispkg in entries:
                    rel_name = full_name[len(pkg.__name__) + 1 :]
                    if any(part.startswith("_") for part in rel_name.split(".")):
                        continue
                    # Leaf packages may hold registrations in their __init__.py and need reloading.
                    if ispkg and full_name in package_children:
                        continue
                    if full_name in owner_modules:
                        continue
                    # Reload only modules that were already loaded before discovery.
                    # This avoids duplicate registrations from package __init__ side-effects.
                    if full_name not in preloaded or full_name not in sys.modules:
                        continue
                    try:
                        importlib.reload(sys.modules[full_name])
                    except ImportError as exc:
                        _warn_or_raise_optional_import(
                            registry_name=self.name,
                            package_name=pkg.__name__,
                            module_name=full_name,
                            exc=exc,
                        )
            return dict(self._items)
        finally:
            self._items = original_items
