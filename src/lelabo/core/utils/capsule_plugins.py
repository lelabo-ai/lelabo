from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import warnings
from pathlib import Path
from typing import Sequence


_SUPPORTED_KINDS = (
    "models",
    "update_rules",
    "datasets",
    "metrics",
    "losses",
    "initializers",
    "schedulers",
    "optimizers",
    "callbacks",
)
_LOADED_BY_FILE: dict[str, str] = {}


def find_active_capsule_root(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "capsule.toml").is_file():
            return candidate
    return None


def _sanitize_token(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", raw)


def _module_name_for(path: Path, kind: str) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    stem = _sanitize_token(path.stem)
    return f"lab_capsule_{kind}_{stem}_{digest}"


def _iter_plugin_files(capsule_root: Path, kind: str) -> list[Path]:
    folder = capsule_root / kind
    if not folder.is_dir():
        return []
    return sorted(
        p.resolve()
        for p in folder.glob("*.py")
        if p.is_file() and p.name != "__init__.py" and not p.name.startswith("_")
    )


def _missing_module_name(exc: BaseException) -> str:
    return str(getattr(exc, "name", "") or "").strip()


def _looks_like_local_capsule_module(capsule_root: Path, missing_name: str) -> bool:
    first = str(missing_name).strip().split(".", 1)[0]
    if not first:
        return False
    candidates = (
        capsule_root / f"{first}.py",
        capsule_root / first / "__init__.py",
        capsule_root / first,
    )
    return any(candidate.exists() for candidate in candidates)


def _is_optional_capsule_dependency(exc: BaseException, *, capsule_root: Path) -> bool:
    if not isinstance(exc, ModuleNotFoundError):
        return False
    missing_name = _missing_module_name(exc)
    if not missing_name:
        return False
    if missing_name == "lelabo" or missing_name.startswith("lelabo."):
        return False
    if _looks_like_local_capsule_module(capsule_root, missing_name):
        return False
    return True


def plugin_files_fingerprint(capsule_root: Path | None, *, kinds: Sequence[str]) -> tuple[tuple[str, tuple[tuple[str, int, int], ...]], ...]:
    """
    Lightweight content fingerprint for plugin files under a capsule root.
    Used to skip expensive registry refreshes when plugin files did not change.
    """
    if capsule_root is None:
        return tuple()

    root = Path(capsule_root).resolve()
    out: list[tuple[str, tuple[tuple[str, int, int], ...]]] = []
    for kind in kinds:
        files = _iter_plugin_files(root, kind)
        rows: list[tuple[str, int, int]] = []
        for path in files:
            try:
                st = path.stat()
                rows.append((str(path), int(st.st_mtime_ns), int(st.st_size)))
            except OSError:
                # If a file disappears between listing and stat, ignore and continue.
                continue
        out.append((str(kind), tuple(rows)))
    return tuple(out)


def _load_plugin(path: Path, kind: str) -> str:
    mod_name = _module_name_for(path, kind)
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build module spec for capsule plugin '{path}'.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod_name


def load_capsule_plugins(
    *,
    kinds: Sequence[str],
    capsule_root: Path | None = None,
) -> list[Path]:
    root = (capsule_root.resolve() if capsule_root else find_active_capsule_root())
    if root is None:
        return []

    loaded: list[Path] = []
    for kind in kinds:
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(f"Unsupported capsule plugin kind '{kind}'. Supported: {_SUPPORTED_KINDS}")

        for path in _iter_plugin_files(root, kind):
            key = str(path)
            if key in _LOADED_BY_FILE:
                continue
            try:
                mod_name = _load_plugin(path, kind)
            except Exception as exc:
                if _is_optional_capsule_dependency(exc, capsule_root=root):
                    warnings.warn(
                        "Skipping capsule plugin due to missing optional dependency: "
                        f"kind='{kind}', plugin='{path}', capsule_root='{root}', "
                        f"missing_dependency='{_missing_module_name(exc)}'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                raise RuntimeError(
                    f"Failed to load capsule plugin kind='{kind}' at '{path}' "
                    f"from capsule '{root}': {exc}"
                ) from exc
            _LOADED_BY_FILE[key] = mod_name
            loaded.append(path)
    return loaded


def load_installed_capsule_plugins(
    *,
    kinds: Sequence[str],
    capsules_dir: Path | None = None,
) -> list[Path]:
    from ...capsule.registry import list_capsules

    loaded: list[Path] = []
    failures: list[tuple[Path, str]] = []
    rows = list_capsules(capsules_dir)
    for row in rows:
        raw_path = str(row.get("path", "")).strip()
        if not raw_path:
            continue
        root = Path(raw_path).expanduser().resolve()
        if not root.exists():
            continue
        try:
            loaded.extend(load_capsule_plugins(kinds=kinds, capsule_root=root))
        except Exception as exc:
            failures.append((root, str(exc)))
            continue

    if failures:
        rendered = "; ".join(f"{path}: {msg}" for path, msg in failures)
        raise RuntimeError(f"Failed to load installed capsule plugins: {rendered}")
    return loaded


def reset_capsule_plugin_cache() -> None:
    for mod_name in list(_LOADED_BY_FILE.values()):
        sys.modules.pop(mod_name, None)
    _LOADED_BY_FILE.clear()


__all__ = [
    "find_active_capsule_root",
    "load_capsule_plugins",
    "load_installed_capsule_plugins",
    "plugin_files_fingerprint",
    "reset_capsule_plugin_cache",
]
