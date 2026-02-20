from __future__ import annotations

from typing import Any, Dict


CAPSULE_SCHEMA_VERSION = "1.0"


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

    capsule_id = str(manifest.get("capsule_id", "")).strip()
    if not capsule_id:
        raise ManifestError("capsule_id must be non-empty.")

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
