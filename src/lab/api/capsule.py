from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..capsule.install import install_capsule as _install_capsule
from ..capsule.pack import pack_capsule as _pack_capsule
from ..capsule.registry import get_capsule as _get_capsule
from ..capsule.registry import list_capsules as _list_capsules
from ..capsule.rerun import rerun_capsule as _rerun_capsule
from ..capsule.create import create_capsule_scaffold as _create_capsule_scaffold
from ..capsule.remove import remove_capsule as _remove_capsule


def pack_capsule(
    *,
    source: Path,
    out_path: Path | None = None,
    capsule_id: str | None = None,
    include_code_snapshot: bool = False,
) -> Path:
    return _pack_capsule(
        source=source,
        out_path=out_path,
        capsule_id=capsule_id,
        include_code_snapshot=include_code_snapshot,
    )


def install_capsule(
    *,
    bundle_path: Path,
    alias: str | None = None,
    capsules_dir: Path | None = None,
) -> dict[str, Any]:
    return _install_capsule(
        bundle_path=bundle_path,
        alias=alias,
        capsules_dir=capsules_dir,
    )


def list_capsules(capsules_dir: Path | None = None) -> list[dict[str, Any]]:
    return _list_capsules(capsules_dir)


def show_capsule(capsule_or_alias: str, capsules_dir: Path | None = None) -> dict[str, Any]:
    row = _get_capsule(capsule_or_alias, capsules_dir)
    if row is None:
        raise ValueError(f"Unknown capsule '{capsule_or_alias}'")
    return row


def rerun_capsule(
    *,
    capsule_or_alias: str,
    capsules_dir: Path | None = None,
    env_mode: str = "current",
    extra_args: Sequence[str] | None = None,
) -> int:
    return int(
        _rerun_capsule(
            capsule_or_alias=capsule_or_alias,
            capsules_dir=capsules_dir,
            env_mode=env_mode,
            extra_args=extra_args,
        )
    )


def create_capsule(
    *,
    capsule_name: str,
    base_dir: Path | None = None,
    force: bool = False,
    register: bool = True,
    alias: str | None = None,
    capsules_dir: Path | None = None,
) -> Path:
    return _create_capsule_scaffold(
        capsule_name=capsule_name,
        base_dir=base_dir,
        force=force,
        register=register,
        alias=alias,
        capsules_dir=capsules_dir,
    )


def remove_capsule(
    *,
    capsule_or_alias: str,
    capsules_dir: Path | None = None,
    delete_files: bool = True,
) -> dict[str, Any]:
    return _remove_capsule(
        capsule_or_alias=capsule_or_alias,
        capsules_dir=capsules_dir,
        delete_files=delete_files,
    )
