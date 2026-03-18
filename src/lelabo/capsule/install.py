"""Install external capsule bundles into the local capsule store."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from .checksums import verify_checksums
from .registry import add_capsule_entry, default_capsules_dir
from .schema import CAPSULE_SCHEMA_VERSION, normalize_capsule_id, validate_manifest


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


def install_capsule_from_directory(
    *,
    source_dir: Path,
    alias: str | None = None,
    capsules_dir: Path | None = None,
    source_bundle: str | None = None,
    source_meta: dict[str, Any] | None = None,
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

        manifest_path = stage / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if source_meta is not None:
                manifest["source"] = dict(source_meta)
            validate_manifest(manifest)
        else:
            manifest = _synth_manifest_from_capsule_toml(stage, source_meta=source_meta)
            validate_manifest(manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        capsule_id = normalize_capsule_id(manifest["capsule_id"])
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
        source_bundle=source_bundle or f"local_path:{source}",
        capsules_dir=root,
    )
    return entry


def install_capsule(
    *,
    bundle_path: Path,
    alias: str | None = None,
    capsules_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = bundle_path.resolve()
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle}")

    root = (capsules_dir or default_capsules_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lelabo_capsule_install_") as td:
        tmp = Path(td)
        _extract_bundle(bundle, tmp)

        manifest_path = tmp / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Invalid capsule: manifest.json is missing.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)

        ok, errs = verify_checksums(tmp)
        if not ok:
            raise ValueError("Checksum verification failed: " + "; ".join(errs))

        capsule_id = normalize_capsule_id(manifest["capsule_id"])

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
        source_bundle=str(bundle),
        capsules_dir=root,
    )
    return entry
