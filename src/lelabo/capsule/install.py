"""Install external capsule bundles into the local capsule store."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import shutil
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from .checksums import verify_checksums
from .registry import add_capsule_entry, default_capsules_dir, get_capsule
from .schema import CAPSULE_SCHEMA_VERSION, normalize_capsule_id, validate_manifest

_FINGERPRINT_IGNORED_PARTS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache"}
_FINGERPRINT_IGNORED_NAMES = {"manifest.json", "checksums.sha256", ".DS_Store"}
_FINGERPRINT_IGNORED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".swp", ".swo"}


def _extract_bundle(bundle_path: Path, dst: Path) -> None:
    name = bundle_path.name.lower()
    if name.endswith(".tar.zst"):
        try:
            import zstandard as zstd  # type: ignore
        except Exception as exc:
            raise RuntimeError("Cannot install .tar.zst capsule without 'zstandard' package.") from exc

        with bundle_path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tf:
                    try:
                        tf.extractall(dst, filter="data")
                    except TypeError:
                        tf.extractall(dst)
        return

    with tarfile.open(bundle_path, mode="r:*") as tf:
        try:
            tf.extractall(dst, filter="data")
        except TypeError:
            tf.extractall(dst)


def _safe_capsule_dst(root: Path, capsule_id: str) -> Path:
    dst = (root / capsule_id).resolve()
    if dst == root or root not in dst.parents:
        raise ValueError(f"Unsafe capsule destination path resolved outside capsules dir: {dst}")
    return dst


def _iter_fingerprint_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _FINGERPRINT_IGNORED_PARTS for part in rel.parts):
            continue
        if rel.name in _FINGERPRINT_IGNORED_NAMES:
            continue
        if p.suffix.lower() in _FINGERPRINT_IGNORED_SUFFIXES:
            continue
        yield p, rel.as_posix()


def compute_capsule_fingerprint(source_dir: Path) -> str:
    source = source_dir.resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Capsule source directory not found: {source}")
    h = hashlib.sha256()
    for p, rel in _iter_fingerprint_files(source):
        h.update(rel.encode("utf-8", errors="replace"))
        h.update(b"\0")
        with p.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _synth_manifest_from_capsule_toml(source_dir: Path, *, source_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    capsule_toml = source_dir / "capsule.toml"
    if not capsule_toml.exists():
        raise ValueError("Invalid capsule source: expected manifest.json or capsule.toml at repo root.")

    raw = tomllib.loads(capsule_toml.read_text(encoding="utf-8"))
    capsule_section = raw.get("capsule", {}) if isinstance(raw, dict) else {}
    if not isinstance(capsule_section, dict):
        capsule_section = {}
    inferred_name = str(capsule_section.get("name", "")).strip() or source_dir.name
    capsule_id = normalize_capsule_id(inferred_name)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "capsule_id": capsule_id,
        "created_at": created_at,
        "kind": "config_only",
        "source": source_meta or {"path": str(source_dir), "type": "directory"},
        "entrypoints": [],
        "artifacts": {"scaffold": True},
        "replay": {"command": None, "cwd": str(source_dir), "tolerance_profile": "relaxed"},
    }


def _load_stage_manifest(stage: Path, *, source_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_path = stage / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if source_meta is not None:
            manifest["source"] = dict(source_meta)
        validate_manifest(manifest)
        return manifest

    manifest = _synth_manifest_from_capsule_toml(stage, source_meta=source_meta)
    validate_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _persist_stage_manifest(stage: Path, manifest: dict[str, Any]) -> None:
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _existing_capsule_fingerprint(existing: dict[str, Any]) -> str | None:
    token = str(existing.get("fingerprint", "") or "").strip()
    if token:
        return token
    path = Path(str(existing.get("path", "") or "")).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return None
    try:
        return compute_capsule_fingerprint(path)
    except Exception:
        return None


def inspect_capsule_directory(
    source_dir: Path,
    *,
    source_meta: dict[str, Any] | None = None,
    capsule_id_override: str | None = None,
) -> dict[str, Any]:
    source = source_dir.resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Capsule source directory not found: {source}")

    with tempfile.TemporaryDirectory(prefix="lelabo_capsule_inspect_") as td:
        stage = Path(td) / "capsule"
        shutil.copytree(
            source,
            stage,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
        )
        manifest = _load_stage_manifest(stage, source_meta=source_meta)
        capsule_id = normalize_capsule_id(capsule_id_override or manifest["capsule_id"])
        fingerprint = compute_capsule_fingerprint(stage)
        return {
            "capsule_id": capsule_id,
            "kind": manifest.get("kind"),
            "fingerprint": fingerprint,
        }


def install_capsule_from_directory(
    *,
    source_dir: Path,
    alias: str | None = None,
    capsules_dir: Path | None = None,
    source_bundle: str | None = None,
    source_meta: dict[str, Any] | None = None,
    capsule_id_override: str | None = None,
    force_replace: bool = False,
) -> dict[str, Any]:
    """Install a capsule directly from a source directory."""
    source = source_dir.resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Capsule source directory not found: {source}")

    root = (capsules_dir or default_capsules_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lelabo_capsule_install_dir_") as td:
        stage = Path(td) / "capsule"
        shutil.copytree(
            source,
            stage,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
        )

        manifest = _load_stage_manifest(stage, source_meta=source_meta)
        capsule_id = normalize_capsule_id(capsule_id_override or manifest["capsule_id"])
        manifest["capsule_id"] = capsule_id
        manifest["fingerprint"] = compute_capsule_fingerprint(stage)
        _persist_stage_manifest(stage, manifest)

        incoming_source_bundle = source_bundle or f"local_path:{source}"
        existing = get_capsule(capsule_id, root)
        if existing is not None and not bool(force_replace):
            existing_fp = _existing_capsule_fingerprint(existing)
            if existing_fp and existing_fp == str(manifest.get("fingerprint", "")):
                out = dict(existing)
                out["install_action"] = "unchanged"
                out["replaced_existing"] = False
                return out
            existing_source = str(existing.get("source_bundle", "") or "").strip()
            if existing_source and existing_source == str(incoming_source_bundle):
                out = dict(existing)
                out["install_action"] = "unchanged"
                out["replaced_existing"] = False
                return out
            raise ValueError(
                f"Capsule id '{capsule_id}' already exists in store at '{existing.get('path')}'. "
                "Use `--force-replace` to replace it, or `--rename-to <new_id>` to install side-by-side."
            )
        dst = _safe_capsule_dst(root, capsule_id)
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.copytree(stage, dst)

    entry = add_capsule_entry(
        capsule_id=capsule_id,
        capsule_path=dst,
        manifest=manifest,
        alias=alias,
        source_bundle=incoming_source_bundle,
        capsules_dir=root,
    )
    entry["install_action"] = "replaced" if bool(force_replace) else "installed"
    entry["replaced_existing"] = bool(force_replace)
    return entry


def install_capsule(
    *,
    bundle_path: Path,
    alias: str | None = None,
    capsules_dir: Path | None = None,
    capsule_id_override: str | None = None,
    force_replace: bool = False,
) -> dict[str, Any]:
    bundle = bundle_path.resolve()
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle}")
    if bundle.is_dir():
        raise ValueError(
            f"Install source '{bundle}' is a directory. "
            "Use directory install through the CLI (`lelabo capsule install <capsule_dir>`) "
            "or pass a capsule bundle archive (.tar.gz/.tar.zst)."
        )

    root = (capsules_dir or default_capsules_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lelabo_capsule_install_") as td:
        tmp = Path(td)
        try:
            _extract_bundle(bundle, tmp)
        except tarfile.TarError as exc:
            raise ValueError(
                f"Invalid capsule bundle '{bundle}'. Expected a valid .tar/.tar.gz/.tar.zst archive."
            ) from exc

        manifest_path = tmp / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Invalid capsule: manifest.json is missing.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)

        ok, errs = verify_checksums(tmp)
        if not ok:
            raise ValueError("Checksum verification failed: " + "; ".join(errs))

        capsule_id = normalize_capsule_id(capsule_id_override or manifest["capsule_id"])
        manifest["capsule_id"] = capsule_id
        manifest["fingerprint"] = compute_capsule_fingerprint(tmp)
        _persist_stage_manifest(tmp, manifest)

        incoming_source_bundle = str(bundle)
        existing = get_capsule(capsule_id, root)
        if existing is not None and not bool(force_replace):
            existing_fp = _existing_capsule_fingerprint(existing)
            if existing_fp and existing_fp == str(manifest.get("fingerprint", "")):
                out = dict(existing)
                out["install_action"] = "unchanged"
                out["replaced_existing"] = False
                return out
            existing_source = str(existing.get("source_bundle", "") or "").strip()
            if existing_source and existing_source == incoming_source_bundle:
                out = dict(existing)
                out["install_action"] = "unchanged"
                out["replaced_existing"] = False
                return out
            raise ValueError(
                f"Capsule id '{capsule_id}' already exists in store at '{existing.get('path')}'. "
                "Use `--force-replace` to replace it, or `--rename-to <new_id>` to install side-by-side."
            )

        dst = _safe_capsule_dst(root, capsule_id)
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.copytree(tmp, dst)

    entry = add_capsule_entry(
        capsule_id=capsule_id,
        capsule_path=dst,
        manifest=manifest,
        alias=alias,
        source_bundle=incoming_source_bundle,
        capsules_dir=root,
    )
    entry["install_action"] = "replaced" if bool(force_replace) else "installed"
    entry["replaced_existing"] = bool(force_replace)
    return entry
