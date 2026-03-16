"""Registry snapshots combining built-ins with capsule-provided exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .discovery import CapsulePluginExport, get_capsule_plugin_exports, load_capsule_plugin_symbol


@dataclass(frozen=True)
class CapsuleRegistrySnapshot:
    registry_name: str
    builtins: dict[str, Any]
    capsule_exports: dict[str, CapsulePluginExport]

    def names(self) -> list[str]:
        return sorted({*self.builtins.keys(), *self.capsule_exports.keys()})

    def get_builtin(self, name: str) -> Any | None:
        return self.builtins.get(str(name).strip().lower())

    def get_export(self, name: str) -> CapsulePluginExport | None:
        return self.capsule_exports.get(str(name).strip().lower())

    def get(self, name: str) -> Any:
        key = str(name).strip().lower()
        if key in self.builtins:
            return self.builtins[key]
        export = self.capsule_exports.get(key)
        if export is not None:
            return load_capsule_plugin_symbol(export)
        raise ValueError(f"[{self.registry_name}] Unknown '{name}'. Available: {self.names()}")


def build_capsule_registry_snapshot(
    *,
    registry_name: str,
    kind: str,
    builtins: dict[str, Any],
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> CapsuleRegistrySnapshot:
    capsule_exports: dict[str, CapsulePluginExport] = {}
    for export in get_capsule_plugin_exports(
        kinds=(kind,),
        capsules_dir=capsules_dir,
        extra_capsule_roots=extra_capsule_roots,
    ):
        key = str(export.name).strip().lower()
        if key in builtins:
            raise RuntimeError(
                f"Capsule plugin '{key}' conflicts with built-in {registry_name} entry '{key}'."
            )
        if key in capsule_exports:
            existing = capsule_exports[key]
            raise RuntimeError(
                f"Duplicate capsule {registry_name} entry '{key}' "
                f"from '{existing.capsule_root}' and '{export.capsule_root}'."
            )
        capsule_exports[key] = export
    return CapsuleRegistrySnapshot(
        registry_name=str(registry_name),
        builtins=dict(builtins),
        capsule_exports=capsule_exports,
    )
