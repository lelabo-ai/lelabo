from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Sequence
import warnings


try:
    from ... import __version__ as LELABO_VERSION
except Exception:  # pragma: no cover - fallback for namespace-stubbed environments.
    LELABO_VERSION = "0.0.0"


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
_INDEX_SCHEMA_VERSION = "1"
_MODULE_CACHE_FINGERPRINTS: dict[str, str] = {}
_IN_MEMORY_INDEXES: dict[str, "CapsulePluginIndex"] = {}

_REGISTRY_SPECS: dict[str, dict[str, Any]] = {
    "models": {
        "module": "lelabo.models.registry",
        "public_modules": ("lelabo.models",),
        "registry_attr": "MODEL_REGISTRY",
        "register_fn": "register_model",
        "metadata_attrs": {},
    },
    "update_rules": {
        "module": "lelabo.update_rules.registry",
        "public_modules": ("lelabo.update_rules",),
        "registry_attr": "UPDATE_RULE_REGISTRY",
        "register_fn": "register_update_rule",
        "metadata_attrs": {},
    },
    "datasets": {
        "module": "lelabo.supervised.datasets.registry",
        "public_modules": ("lelabo.supervised.datasets",),
        "registry_attr": "DATASET_REGISTRY",
        "register_fn": "register_dataset",
        "metadata_attrs": {},
    },
    "metrics": {
        "module": "lelabo.metrics.registry",
        "public_modules": ("lelabo.metrics",),
        "registry_attr": "METRIC_REGISTRY",
        "register_fn": "register_metric",
        "metadata_attrs": {
            "metric_kind": "__metric_kind__",
            "metric_params": "__metric_params__",
        },
    },
    "losses": {
        "module": "lelabo.losses.registry",
        "public_modules": ("lelabo.losses",),
        "registry_attr": "LOSS_REGISTRY",
        "register_fn": "register_loss",
        "metadata_attrs": {},
    },
    "initializers": {
        "module": "lelabo.initializers.registry",
        "public_modules": ("lelabo.initializers",),
        "registry_attr": "INITIALIZER_REGISTRY",
        "register_fn": "register_initializer",
        "metadata_attrs": {},
    },
    "schedulers": {
        "module": "lelabo.schedulers.registry",
        "public_modules": ("lelabo.schedulers",),
        "registry_attr": "SCHEDULER_REGISTRY",
        "register_fn": "register_scheduler",
        "metadata_attrs": {},
    },
    "optimizers": {
        "module": "lelabo.optimizers.registry",
        "public_modules": ("lelabo.optimizers",),
        "registry_attr": "OPTIMIZER_REGISTRY",
        "register_fn": "register_optimizer",
        "metadata_attrs": {},
    },
    "callbacks": {
        "module": "lelabo.callbacks.registry",
        "public_modules": ("lelabo.callbacks",),
        "registry_attr": "CALLBACK_REGISTRY",
        "register_fn": "register_callback",
        "metadata_attrs": {},
    },
}


@dataclass(frozen=True)
class CapsulePluginExport:
    kind: str
    name: str
    module: str
    symbol: str
    metadata: dict[str, Any]
    file: str
    capsule_root: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, capsule_root: str | None = None) -> "CapsulePluginExport":
        return cls(
            kind=str(raw.get("kind", "")).strip(),
            name=str(raw.get("name", "")).strip().lower(),
            module=str(raw.get("module", "")).strip(),
            symbol=str(raw.get("symbol", "")).strip(),
            metadata=dict(raw.get("metadata", {}) or {}),
            file=str(raw.get("file", "")).strip(),
            capsule_root=str(capsule_root or raw.get("capsule_root", "")).strip(),
        )


@dataclass(frozen=True)
class CapsulePluginIndex:
    schema_version: str
    capsule_root: str
    lelabo_version: str
    fingerprint: str
    generated_at: str
    exports: tuple[CapsulePluginExport, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CapsulePluginIndex":
        capsule_root = str(raw.get("capsule_root", "")).strip()
        exports = tuple(
            CapsulePluginExport.from_dict(item, capsule_root=capsule_root)
            for item in list(raw.get("exports", []) or [])
            if isinstance(item, dict)
        )
        return cls(
            schema_version=str(raw.get("schema_version", "")).strip(),
            capsule_root=capsule_root,
            lelabo_version=str(raw.get("lelabo_version", "")).strip(),
            fingerprint=str(raw.get("fingerprint", "")).strip(),
            generated_at=str(raw.get("generated_at", "")).strip(),
            exports=exports,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "capsule_root": str(self.capsule_root),
            "lelabo_version": str(self.lelabo_version),
            "fingerprint": str(self.fingerprint),
            "generated_at": str(self.generated_at),
            "exports": [asdict(item) for item in self.exports],
        }


def find_active_capsule_root(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "capsule.toml").is_file():
            return candidate
    return None


def _sanitize_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", str(raw).strip())
    if not token:
        token = "_"
    if token[0].isdigit():
        token = f"_{token}"
    return token


def _capsule_package_name(capsule_root: Path) -> str:
    digest = hashlib.sha1(str(capsule_root.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"lab_capsule_pkg_{digest}"


def _module_name_for(path: Path, kind: str, capsule_root: Path) -> str:
    _ = kind
    rel = path.resolve().relative_to(capsule_root.resolve()).with_suffix("")
    parts = [_sanitize_token(part) for part in rel.parts]
    return ".".join([_capsule_package_name(capsule_root), *parts])


def _skip_walk_dir(name: str) -> bool:
    token = str(name).strip()
    lowered = token.lower()
    if not token:
        return False
    if token.startswith("."):
        return True
    return lowered in {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "build",
        "dist",
        "artifacts",
        "outputs",
    }


def _iter_capsule_python_files(capsule_root: Path) -> list[Path]:
    root = capsule_root.resolve()
    out: list[Path] = []
    for cur_root, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [name for name in dirnames if not _skip_walk_dir(name)]
        cur = Path(cur_root)
        for filename in filenames:
            if not str(filename).endswith(".py"):
                continue
            if str(filename).startswith("."):
                continue
            path = (cur / filename).resolve()
            if path.is_file():
                out.append(path)
    return sorted(out)


def _iter_plugin_files(capsule_root: Path, kind: str) -> list[Path]:
    folder = capsule_root / kind
    if not folder.is_dir():
        return []
    return sorted(
        p.resolve()
        for p in folder.glob("*.py")
        if p.is_file() and p.name != "__init__.py" and not p.name.startswith("_")
    )


def _python_file_content_rows(paths: Sequence[Path], *, root: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in paths:
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        rel = str(path.resolve().relative_to(root.resolve())).replace(os.sep, "/")
        rows.append((rel, hashlib.sha256(payload).hexdigest()))
    return rows


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


def _user_cache_root() -> Path:
    raw = os.getenv("LELABO_PLUGIN_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()

    home = Path.home()
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(home / "AppData" / "Local")
        return (Path(base).expanduser().resolve() / "LeLabo" / "cache" / "plugin_index").resolve()
    if sys.platform == "darwin":
        return (home / "Library" / "Caches" / "lelabo" / "plugin_index").resolve()
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg:
        return (Path(xdg).expanduser().resolve() / "lelabo" / "plugin_index").resolve()
    return (home / ".cache" / "lelabo" / "plugin_index").resolve()


def _resolved_cache_root() -> Path:
    preferred = _user_cache_root()
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        fallback = (Path(tempfile.gettempdir()) / "lelabo" / "plugin_index").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _index_cache_path(capsule_root: Path) -> Path:
    root = capsule_root.resolve()
    key = hashlib.sha1(f"{LELABO_VERSION}::{root}".encode("utf-8")).hexdigest()
    return _resolved_cache_root() / f"{key}.json"


def _pycache_prefix_for_capsule(capsule_root: Path, fingerprint: str) -> Path:
    root = capsule_root.resolve()
    key = hashlib.sha1(f"{root}::{fingerprint}".encode("utf-8")).hexdigest()
    return _resolved_cache_root() / "pycache" / key


def fingerprint_capsule_source(capsule_root: Path) -> str:
    root = capsule_root.resolve()
    rows = _python_file_content_rows(_iter_capsule_python_files(root), root=root)
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plugin_files_fingerprint(
    capsule_root: Path | None,
    *,
    kinds: Sequence[str],
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    if capsule_root is None:
        return tuple()

    root = Path(capsule_root).resolve()
    out: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for kind in kinds:
        files = _iter_plugin_files(root, kind)
        rows = tuple(_python_file_content_rows(files, root=root))
        out.append((str(kind), rows))
    return tuple(out)


def _read_index_file(path: Path) -> CapsulePluginIndex | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    index = CapsulePluginIndex.from_dict(raw)
    if index.schema_version != _INDEX_SCHEMA_VERSION:
        return None
    if index.lelabo_version != str(LELABO_VERSION):
        return None
    return index


def _write_index_file(path: Path, index: CapsulePluginIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(index.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _source_root_for_subprocess() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_index_subprocess(capsule_root: Path, *, fingerprint: str) -> tuple[CapsulePluginIndex, list[str]]:
    worker_module = "lelabo.core.utils.capsule_plugin_worker"
    env = dict(os.environ)
    source_root = str(_source_root_for_subprocess())
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = source_root if not existing_pythonpath else f"{source_root}{os.pathsep}{existing_pythonpath}"
    env["PYTHONPYCACHEPREFIX"] = str(_pycache_prefix_for_capsule(capsule_root, fingerprint))
    proc = subprocess.run(
        [sys.executable, "-m", worker_module, "--capsule-root", str(capsule_root.resolve())],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        raise RuntimeError(
            f"Failed to index capsule plugins for '{capsule_root.resolve()}': {detail}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive.
        raise RuntimeError(
            f"Failed to parse capsule plugin index output for '{capsule_root.resolve()}': {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Capsule plugin index output for '{capsule_root.resolve()}' is not a JSON object.")
    warnings_out = [str(item) for item in list(payload.get("warnings", []) or []) if str(item).strip()]
    index = CapsulePluginIndex.from_dict(payload)
    if index.schema_version != _INDEX_SCHEMA_VERSION:
        raise RuntimeError(
            f"Capsule plugin index for '{capsule_root.resolve()}' returned unsupported schema '{index.schema_version}'."
        )
    return index, warnings_out


def ensure_capsule_plugin_index(capsule_root: Path) -> CapsulePluginIndex:
    root = capsule_root.resolve()
    current_fingerprint = fingerprint_capsule_source(root)
    cache_path = _index_cache_path(root)
    memory_key = str(root)

    cached = _IN_MEMORY_INDEXES.get(memory_key)
    if cached is not None and cached.fingerprint == current_fingerprint:
        return cached

    disk_index = _read_index_file(cache_path)
    if (
        disk_index is not None
        and disk_index.capsule_root == str(root)
        and disk_index.fingerprint == current_fingerprint
    ):
        _IN_MEMORY_INDEXES[memory_key] = disk_index
        return disk_index

    index, warnings_out = _run_index_subprocess(root, fingerprint=current_fingerprint)
    if index.capsule_root != str(root):
        raise RuntimeError(
            f"Capsule plugin index root mismatch for '{root}': got '{index.capsule_root}'."
        )
    if index.fingerprint != current_fingerprint:
        # If source changed during indexing, prefer the fresh index from the worker.
        current_fingerprint = str(index.fingerprint)
    for msg in warnings_out:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    _write_index_file(cache_path, index)
    _IN_MEMORY_INDEXES[memory_key] = index
    return index


def _normalize_extra_roots(extra_capsule_roots: Sequence[Path] | None) -> tuple[Path, ...]:
    roots = {Path(root).resolve() for root in list(extra_capsule_roots or [])}
    return tuple(sorted(roots, key=str))


def _installed_capsule_roots(capsules_dir: Path | None = None) -> list[Path]:
    from ...capsule.registry import list_capsules

    roots: list[Path] = []
    for row in list_capsules(capsules_dir):
        raw_path = str(row.get("path", "")).strip()
        if not raw_path:
            continue
        root = Path(raw_path).expanduser().resolve()
        if root.exists():
            roots.append(root)
    return roots


def _active_and_explicit_roots(extra_capsule_roots: Sequence[Path] | None = None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    active_root = find_active_capsule_root()
    for root in [active_root, *_normalize_extra_roots(extra_capsule_roots)]:
        if root is None:
            continue
        resolved = root.resolve()
        token = str(resolved)
        if token in seen:
            continue
        seen.add(token)
        out.append(resolved)
    return out


def get_capsule_plugin_exports(
    *,
    kinds: Sequence[str],
    capsules_dir: Path | None = None,
    extra_capsule_roots: Sequence[Path] | None = None,
) -> list[CapsulePluginExport]:
    normalized_kinds = tuple(str(kind).strip() for kind in kinds if str(kind).strip())
    for kind in normalized_kinds:
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(f"Unsupported capsule plugin kind '{kind}'. Supported: {_SUPPORTED_KINDS}")

    exports_by_name: dict[tuple[str, str], CapsulePluginExport] = {}
    seen_roots = {str(root) for root in _active_and_explicit_roots(extra_capsule_roots)}
    for root in _active_and_explicit_roots(extra_capsule_roots):
        index = ensure_capsule_plugin_index(root)
        for export in index.exports:
            if export.kind not in normalized_kinds:
                continue
            key = (str(export.kind), str(export.name).lower())
            if key in exports_by_name:
                existing = exports_by_name[key]
                raise RuntimeError(
                    f"Duplicate capsule plugin '{export.name}' for kind '{export.kind}' "
                    f"from '{existing.capsule_root}' and '{export.capsule_root}'."
                )
            exports_by_name[key] = export

    failures: list[tuple[Path, str]] = []
    for root in _installed_capsule_roots(capsules_dir):
        if str(root) in seen_roots:
            continue
        try:
            index = ensure_capsule_plugin_index(root)
        except Exception as exc:
            failures.append((root, str(exc)))
            continue
        for export in index.exports:
            if export.kind not in normalized_kinds:
                continue
            key = (str(export.kind), str(export.name).lower())
            if key in exports_by_name:
                existing = exports_by_name[key]
                raise RuntimeError(
                    f"Duplicate capsule plugin '{export.name}' for kind '{export.kind}' "
                    f"from '{existing.capsule_root}' and '{export.capsule_root}'."
                )
            exports_by_name[key] = export

    if failures:
        rendered = "; ".join(f"{path}: {msg}" for path, msg in failures)
        raise RuntimeError(f"Failed to load installed capsule plugins: {rendered}")

    return sorted(
        exports_by_name.values(),
        key=lambda item: (str(item.kind), str(item.name), str(item.capsule_root)),
    )


@contextmanager
def _temporary_sys_path(path: Path):
    token = str(path.resolve())
    sys.path.insert(0, token)
    try:
        yield
    finally:
        try:
            sys.path.remove(token)
        except ValueError:
            pass


@contextmanager
def _temporary_pycache_prefix(path: Path):
    original = getattr(sys, "pycache_prefix", None)
    target = str(path.resolve())
    Path(target).mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = target
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.pycache_prefix = original
        importlib.invalidate_caches()


def _ensure_namespace_packages(capsule_root: Path, file_path: Path) -> None:
    root = capsule_root.resolve()
    rel = file_path.resolve().relative_to(root)
    package_name = _capsule_package_name(root)
    packages: list[tuple[str, Path]] = [(package_name, root)]

    if len(rel.parts) > 1:
        cur_name = package_name
        cur_path = root
        for part in rel.parts[:-1]:
            cur_name = f"{cur_name}.{_sanitize_token(part)}"
            cur_path = cur_path / part
            packages.append((cur_name, cur_path))

    for name, pkg_path in packages:
        mod = sys.modules.get(name)
        if mod is None:
            mod = ModuleType(name)
            mod.__package__ = name
            mod.__path__ = [str(pkg_path.resolve())]  # type: ignore[attr-defined]
            sys.modules[name] = mod
            continue
        mod.__package__ = name
        mod.__path__ = [str(pkg_path.resolve())]  # type: ignore[attr-defined]


def _noop_register(*args, **kwargs):
    _ = (args, kwargs)

    def _decorator(obj):
        return obj

    return _decorator


@contextmanager
def suspend_capsule_registration():
    originals: list[tuple[Any, str, Any]] = []
    for spec in _REGISTRY_SPECS.values():
        mod = importlib.import_module(str(spec["module"]))
        fn_name = str(spec["register_fn"])
        if hasattr(mod, fn_name):
            originals.append((mod, fn_name, getattr(mod, fn_name)))
            setattr(mod, fn_name, _noop_register)

        registry = getattr(mod, str(spec["registry_attr"]), None)
        if registry is not None and hasattr(registry, "register"):
            originals.append((registry, "register", getattr(registry, "register")))
            setattr(registry, "register", _noop_register)

        for public_module_name in tuple(spec.get("public_modules", ()) or ()):
            try:
                public_mod = importlib.import_module(str(public_module_name))
            except Exception:
                continue
            if hasattr(public_mod, fn_name):
                originals.append((public_mod, fn_name, getattr(public_mod, fn_name)))
                setattr(public_mod, fn_name, _noop_register)
    try:
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)


def _clear_capsule_modules(capsule_root: Path) -> None:
    package_name = _capsule_package_name(capsule_root)
    for mod_name in [name for name in list(sys.modules.keys()) if name == package_name or name.startswith(f"{package_name}.")]:
        sys.modules.pop(mod_name, None)


def _load_capsule_module(
    path: Path,
    *,
    kind: str,
    capsule_root: Path,
    allow_registration: bool,
    fingerprint: str | None = None,
) -> ModuleType:
    module_name = _module_name_for(path, kind, capsule_root)
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    _ensure_namespace_packages(capsule_root, path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build module spec for capsule plugin '{path}'.")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        pycache_fingerprint = str(fingerprint or fingerprint_capsule_source(capsule_root))
        with _temporary_pycache_prefix(_pycache_prefix_for_capsule(capsule_root, pycache_fingerprint)):
            with _temporary_sys_path(capsule_root):
                if allow_registration:
                    spec.loader.exec_module(mod)
                else:
                    with suspend_capsule_registration():
                        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def load_capsule_plugin_symbol(export: CapsulePluginExport) -> Any:
    root = Path(export.capsule_root).resolve()
    index = ensure_capsule_plugin_index(root)
    current = next(
        (
            item
            for item in index.exports
            if item.kind == export.kind and item.name == export.name
        ),
        None,
    )
    if current is None:
        raise RuntimeError(
            f"Capsule plugin '{export.name}' ({export.kind}) is no longer present in '{root}'."
        )

    root_key = str(root)
    if _MODULE_CACHE_FINGERPRINTS.get(root_key) != index.fingerprint:
        _clear_capsule_modules(root)
        _MODULE_CACHE_FINGERPRINTS[root_key] = str(index.fingerprint)

    file_path = (root / current.file).resolve()
    mod = _load_capsule_module(
        file_path,
        kind=str(current.kind),
        capsule_root=root,
        allow_registration=False,
        fingerprint=str(index.fingerprint),
    )
    if not hasattr(mod, current.symbol):
        raise RuntimeError(
            f"Capsule plugin module '{current.module}' does not define symbol '{current.symbol}'."
        )
    return getattr(mod, current.symbol)


def load_capsule_plugins(
    *,
    kinds: Sequence[str],
    capsule_root: Path | None = None,
) -> list[Path]:
    root = (capsule_root.resolve() if capsule_root else find_active_capsule_root())
    if root is None:
        return []
    index = ensure_capsule_plugin_index(root)
    wanted = {str(kind).strip() for kind in kinds if str(kind).strip()}
    paths = {
        (Path(index.capsule_root) / export.file).resolve()
        for export in index.exports
        if export.kind in wanted
    }
    return sorted(paths)


def load_installed_capsule_plugins(
    *,
    kinds: Sequence[str],
    capsules_dir: Path | None = None,
) -> list[Path]:
    loaded: set[Path] = set()
    failures: list[tuple[Path, str]] = []
    wanted = {str(kind).strip() for kind in kinds if str(kind).strip()}
    for root in _installed_capsule_roots(capsules_dir):
        try:
            index = ensure_capsule_plugin_index(root)
        except Exception as exc:
            failures.append((root, str(exc)))
            continue
        for export in index.exports:
            if export.kind not in wanted:
                continue
            loaded.add((root / export.file).resolve())
    if failures:
        rendered = "; ".join(f"{path}: {msg}" for path, msg in failures)
        raise RuntimeError(f"Failed to load installed capsule plugins: {rendered}")
    return sorted(loaded)


def reset_capsule_plugin_cache() -> None:
    for root_str in list(_MODULE_CACHE_FINGERPRINTS.keys()):
        _clear_capsule_modules(Path(root_str))
    _MODULE_CACHE_FINGERPRINTS.clear()
    _IN_MEMORY_INDEXES.clear()


__all__ = [
    "CapsulePluginExport",
    "CapsulePluginIndex",
    "ensure_capsule_plugin_index",
    "find_active_capsule_root",
    "fingerprint_capsule_source",
    "get_capsule_plugin_exports",
    "load_capsule_plugin_symbol",
    "load_capsule_plugins",
    "load_installed_capsule_plugins",
    "plugin_files_fingerprint",
    "reset_capsule_plugin_cache",
    "suspend_capsule_registration",
]
