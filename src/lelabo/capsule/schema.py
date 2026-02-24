from __future__ import annotations

import re
from typing import Any, Dict


CAPSULE_SCHEMA_VERSION = "1.0"
_CAPSULE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


_REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "capsule_id",
    "created_at",
    "kind",
    "source",
    "entrypoints",
    "artifacts",
    "replay",
}


class ManifestError(ValueError):
    pass


def _expect_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must be a JSON object.")
    return value


def _expect_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{path} must be a JSON array.")
    return value


def normalize_capsule_id(raw: Any, *, field_name: str = "capsule_id") -> str:
    capsule_id = str(raw).strip()
    if not capsule_id:
        raise ManifestError(f"{field_name} must be non-empty.")
    if capsule_id in {".", ".."}:
        raise ManifestError(f"{field_name} cannot be '.' or '..'.")
    if "/" in capsule_id or "\\" in capsule_id:
        raise ManifestError(f"{field_name} cannot contain path separators.")
    if not _CAPSULE_ID_RE.fullmatch(capsule_id):
        raise ManifestError(f"{field_name} must match [A-Za-z0-9._-]+.")
    return capsule_id


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    _expect_dict(manifest, "manifest")

    missing = sorted(k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in manifest)
    if missing:
        raise ManifestError(f"Missing required manifest keys: {missing}")

    schema_version = str(manifest.get("schema_version", "")).strip()
    if schema_version != CAPSULE_SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported schema_version='{schema_version}'. Expected '{CAPSULE_SCHEMA_VERSION}'."
        )

    normalize_capsule_id(manifest.get("capsule_id", ""))
    if "lelabo_version" in manifest:
        if not str(manifest.get("lelabo_version", "")).strip():
            raise ManifestError("lelabo_version must be non-empty when provided.")

    kind = str(manifest.get("kind", "")).strip().lower()
    if kind not in {"single_run", "sweep", "config_only"}:
        raise ManifestError("kind must be one of: single_run, sweep, config_only.")

    source = _expect_dict(manifest.get("source"), "source")
    if not str(source.get("path", "")).strip():
        raise ManifestError("source.path must be non-empty.")

    entrypoints = _expect_list(manifest.get("entrypoints"), "entrypoints")
    for i, ep in enumerate(entrypoints):
        epd = _expect_dict(ep, f"entrypoints[{i}]")
        cmd = _expect_list(epd.get("cmd", []), f"entrypoints[{i}].cmd")
        if not cmd:
            raise ManifestError(f"entrypoints[{i}].cmd must be non-empty.")

    _expect_dict(manifest.get("artifacts"), "artifacts")
    replay = _expect_dict(manifest.get("replay"), "replay")
    if replay:
        # command is optional for config-only capsules
        cmd = replay.get("command")
        if cmd is not None:
            arr = _expect_list(cmd, "replay.command")
            if not arr:
                raise ManifestError("replay.command must be non-empty when provided.")

    return manifest
