from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys

from .discovery import (
    _INDEX_SCHEMA_VERSION,
    _REGISTRY_SPECS,
    _is_optional_capsule_dependency,
    _iter_plugin_files,
    _load_capsule_module,
    _missing_module_name,
    CapsulePluginExport,
    CapsulePluginIndex,
    LELABO_VERSION,
    fingerprint_capsule_source,
)


def _extract_metadata(kind: str, builder) -> dict[str, object]:
    spec = _REGISTRY_SPECS[str(kind)]
    metadata: dict[str, object] = {}
    for key, attr_name in dict(spec.get("metadata_attrs", {}) or {}).items():
        value = getattr(builder, str(attr_name), None)
        if value is None:
            continue
        if isinstance(value, tuple):
            metadata[str(key)] = list(value)
            continue
        metadata[str(key)] = value
    return metadata


def _relative_file(root: Path, module_name: str) -> str:
    mod = sys.modules.get(module_name)
    file_path = getattr(mod, "__file__", None)
    if not isinstance(file_path, str) or not file_path.strip():
        raise RuntimeError(f"Registered plugin module '{module_name}' has no source file.")
    path = Path(file_path).resolve()
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError as exc:
        raise RuntimeError(
            f"Registered plugin module '{module_name}' is outside capsule root '{root}'."
        ) from exc


def _format_plugin_load_error(*, kind: str, path: Path, capsule_root: Path, exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    register_fn = str(_REGISTRY_SPECS[str(kind)].get("register_fn", "")).strip()
    if isinstance(exc, NameError) and register_fn:
        missing_name = str(getattr(exc, "name", "") or "").strip()
        if missing_name == register_fn:
            detail = (
                f"{detail}. The plugin enables '@{register_fn}(...)' but does not import "
                f"'{register_fn}'. Add it to the imports in '{path.name}' or comment the decorator out."
            )
    return (
        f"Failed to load capsule plugin kind='{kind}' at '{path}' from capsule '{capsule_root}': {detail}"
    )


def build_capsule_plugin_index(capsule_root: Path) -> tuple[CapsulePluginIndex, list[str]]:
    root = capsule_root.resolve()
    current_fingerprint = str(fingerprint_capsule_source(root))
    warnings_out: list[str] = []
    registry_state: dict[str, tuple[object, dict[str, object]]] = {}

    for kind, spec in _REGISTRY_SPECS.items():
        mod = importlib.import_module(str(spec["module"]))
        registry = getattr(mod, str(spec["registry_attr"]))
        builtins = registry.snapshot_discovered_items()
        registry._items = dict(builtins)
        registry_state[kind] = (registry, dict(builtins))

    for kind in _REGISTRY_SPECS:
        for path in _iter_plugin_files(root, kind):
            try:
                _load_capsule_module(
                    path,
                    kind=kind,
                    capsule_root=root,
                    allow_registration=True,
                    fingerprint=current_fingerprint,
                )
            except Exception as exc:
                if _is_optional_capsule_dependency(exc, capsule_root=root):
                    warnings_out.append(
                        "Skipping capsule plugin due to missing optional dependency: "
                        f"kind='{kind}', plugin='{path}', capsule_root='{root}', "
                        f"missing_dependency='{_missing_module_name(exc)}'."
                    )
                    continue
                raise RuntimeError(
                    _format_plugin_load_error(
                        kind=kind,
                        path=path,
                        capsule_root=root,
                        exc=exc,
                    )
                ) from exc

    exports: list[CapsulePluginExport] = []
    for kind, (registry, builtins) in registry_state.items():
        builtin_names = {str(name).strip().lower() for name in builtins.keys()}
        for name, builder in dict(getattr(registry, "_items", {}) or {}).items():
            key = str(name).strip().lower()
            if key in builtin_names:
                continue
            module_name = str(getattr(builder, "__module__", "")).strip()
            symbol = str(getattr(builder, "__name__", "")).strip()
            if not module_name or not symbol:
                raise RuntimeError(
                    f"Registered capsule plugin '{key}' ({kind}) does not expose a stable module/symbol origin."
                )
            exports.append(
                CapsulePluginExport(
                    kind=str(kind),
                    name=key,
                    module=module_name,
                    symbol=symbol,
                    metadata=_extract_metadata(kind, builder),
                    file=_relative_file(root, module_name),
                    capsule_root=str(root),
                )
            )

    exports.sort(key=lambda item: (item.kind, item.name, item.module, item.symbol))
    index = CapsulePluginIndex(
        schema_version=_INDEX_SCHEMA_VERSION,
        capsule_root=str(root),
        lelabo_version=str(LELABO_VERSION),
        fingerprint=current_fingerprint,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        exports=tuple(exports),
    )
    return index, warnings_out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lelabo.capsule-plugin-worker")
    parser.add_argument("--capsule-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        index, warnings_out = build_capsule_plugin_index(Path(args.capsule_root))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = index.to_dict()
    payload["warnings"] = warnings_out
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
