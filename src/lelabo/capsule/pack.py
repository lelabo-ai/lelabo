from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path

from .checksums import write_checksums
from .collect import collect_capsule


def _sanitize_capsule_id(raw: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(raw).strip())
    s = s.strip("-_")
    return s or "capsule"


def _default_exports_dir() -> Path:
    return (Path.cwd() / "outputs" / "exports" / "capsules").resolve()


def _write_tar_from_dir(source_dir: Path, out_path: Path) -> None:
    suffix = out_path.name.lower()
    if suffix.endswith(".tar.zst"):
        try:
            import zstandard as zstd  # type: ignore
        except Exception as exc:
            raise RuntimeError("'.tar.zst' requires 'zstandard' package to be installed.") from exc

        cctx = zstd.ZstdCompressor(level=10)
        with out_path.open("wb") as fh:
            with cctx.stream_writer(fh) as zfh:
                with tarfile.open(fileobj=zfh, mode="w|") as tf:
                    for p in sorted(source_dir.rglob("*")):
                        tf.add(p, arcname=p.relative_to(source_dir).as_posix())
        return

    mode = "w:gz"
    if suffix.endswith(".tar"):
        mode = "w"
    with tarfile.open(out_path, mode=mode) as tf:
        for p in sorted(source_dir.rglob("*")):
            tf.add(p, arcname=p.relative_to(source_dir).as_posix())


def pack_capsule(
    *,
    source: Path,
    out_path: Path | None = None,
    capsule_id: str | None = None,
    include_code_snapshot: bool = False,
) -> Path:
    source = source.resolve()
    cap_id = _sanitize_capsule_id(capsule_id or source.name)

    if out_path is None:
        out_dir = _default_exports_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cap_id}.tar.gz"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lelabo_capsule_stage_") as td:
        stage = Path(td)
        collect_capsule(
            source=source,
            stage_dir=stage,
            capsule_id=cap_id,
            include_code_snapshot=include_code_snapshot,
        )
        write_checksums(stage, exclude={"checksums.sha256"})
        _write_tar_from_dir(stage, out_path)

    return out_path
