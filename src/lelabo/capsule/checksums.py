"""Checksum helpers for capsule bundle integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def write_checksums(root: Path, *, checksums_path: Path | None = None, exclude: set[str] | None = None) -> Path:
    target = checksums_path or (root / "checksums.sha256")
    excludes = exclude or set()

    lines: list[str] = []
    for p in iter_files(root):
        rel = p.relative_to(root).as_posix()
        if rel in excludes:
            continue
        lines.append(f"{sha256_file(p)}  {rel}")

    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def verify_checksums(root: Path, checksums_path: Path | None = None) -> tuple[bool, list[str]]:
    target = checksums_path or (root / "checksums.sha256")
    if not target.exists():
        return False, ["checksums.sha256 missing"]

    errors: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            expected, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"Malformed checksum line: {line}")
            continue
        p = root / rel
        if not p.exists():
            errors.append(f"Missing file: {rel}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            errors.append(f"Checksum mismatch: {rel}")

    return len(errors) == 0, errors
