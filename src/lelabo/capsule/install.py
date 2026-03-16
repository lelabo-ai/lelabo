"""Install external capsule bundles into the local capsule store."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .checksums import verify_checksums
from .registry import add_capsule_entry, default_capsules_dir
from .schema import normalize_capsule_id, validate_manifest


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
