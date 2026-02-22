from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .registry import add_capsule_entry, default_capsules_dir
from .schema import CAPSULE_SCHEMA_VERSION, validate_manifest

_CAPSULE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SCAFFOLD_DIRS = (
    "models",
    "update_rules",
    "datasets",
    "metrics",
    "configs",
    "runs",
)

_EXAMPLE_TEMPLATE_MAP = {
    "models/example.py": "models_example.py",
    "update_rules/example.py": "update_rules_example.py",
    "datasets/example.py": "datasets_example.py",
    "metrics/example.py": "metrics_example.py",
    "configs/example.py": "configs_example.py",
    "runs/example.py": "runs_example.py",
}


def _normalize_capsule_name(raw: str) -> str:
    name = str(raw).strip()
    if not name:
        raise ValueError("Capsule name cannot be empty.")
    if name in {".", ".."}:
        raise ValueError("Capsule name cannot be '.' or '..'.")
    if "/" in name or "\\" in name:
        raise ValueError("Capsule name cannot contain path separators.")
    if not _CAPSULE_NAME_RE.fullmatch(name):
        raise ValueError("Capsule name must match [A-Za-z0-9._-]+.")
    return name


def _readme_template(name: str) -> str:
    return (
        f"# {name}\n\n"
        "LeLabo capsule scaffold.\n\n"
        "## Structure\n\n"
        "- `models/`: custom models\n"
        "- `update_rules/`: custom learning rules\n"
        "- `datasets/`: local dataset helpers\n"
        "- `metrics/`: local training metrics\n"
        "- `configs/`: experiment configs\n"
        "- `runs/`: local run artifacts\n"
        "\n"
        "Each folder contains an `example.py` template you can adapt.\n"
        "\n"
        "## Local Registries\n\n"
        "When you run `lelabo` inside this capsule (or a subfolder), LeLabo auto-loads\n"
        "`models/*.py`, `update_rules/*.py`, `datasets/*.py`, and `metrics/*.py`.\n"
        "Use the standard decorators in these files:\n\n"
        "- `from lab.models.registry import register_model`\n"
        "- `from lab.update_rules.registry import register_update_rule`\n"
        "- `from lab.supervised.datasets.registry import register_dataset`\n"
        "- `from lab.metrics.registry import register_metric`\n"
    )


def _capsule_toml_template(name: str) -> str:
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        "[capsule]\n"
        f'name = "{name}"\n'
        f'created_at = "{created_at}"\n'
        'format = "lelabo.capsule.scaffold.v1"\n'
    )


def _manifest_template(name: str, target: Path) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "capsule_id": name,
        "created_at": created_at,
        "kind": "config_only",
        "source": {
            "path": str(target),
            "type": "directory",
        },
        "entrypoints": [],
        "artifacts": {
            "scaffold": True,
        },
        "replay": {
            "command": None,
            "cwd": str(target),
            "tolerance_profile": "relaxed",
        },
    }


@lru_cache(maxsize=1)
def _load_example_templates() -> dict[str, str]:
    pkg = resources.files("lab.capsule.templates")
    out: dict[str, str] = {}
    for rel_path, src_name in _EXAMPLE_TEMPLATE_MAP.items():
        src = pkg.joinpath(src_name)
        out[rel_path] = src.read_text(encoding="utf-8")
    return out


def _template_files(name: str) -> dict[str, str]:
    templates = {
        "README.md": _readme_template(name),
        "capsule.toml": _capsule_toml_template(name),
        "models/__init__.py": '"""Custom models for this capsule."""\n',
        "update_rules/__init__.py": '"""Custom update rules for this capsule."""\n',
        "datasets/__init__.py": '"""Custom datasets for this capsule."""\n',
        "metrics/__init__.py": '"""Custom metrics for this capsule."""\n',
        "configs/README.md": "# Configs\n\nPlace experiment config files here.\n",
    }
    templates.update(_load_example_templates())
    return templates


def create_capsule_scaffold(
    *,
    capsule_name: str,
    base_dir: Path | None = None,
    force: bool = False,
    register: bool = True,
    alias: str | None = None,
    capsules_dir: Path | None = None,
) -> Path:
    name = _normalize_capsule_name(capsule_name)
    root = (base_dir or Path.cwd()).resolve()
    target = (root / name).resolve()

    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"Cannot create capsule at '{target}': path exists and is not a directory.")
        if not force:
            raise FileExistsError(f"Capsule directory already exists: {target}. Use --force to continue.")
    else:
        target.mkdir(parents=True, exist_ok=False)

    for rel_dir in _SCAFFOLD_DIRS:
        path = target / rel_dir
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"Cannot create directory '{path}': path exists and is not a directory.")
        path.mkdir(parents=True, exist_ok=True)

    for rel_file, content in _template_files(name).items():
        path = target / rel_file
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _manifest_template(name, target)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    validate_manifest(manifest)

    if register:
        registry_root = (capsules_dir or default_capsules_dir()).resolve()
        add_capsule_entry(
            capsule_id=name,
            capsule_path=target,
            manifest=manifest,
            alias=alias or name,
            source_bundle="local_scaffold",
            capsules_dir=registry_root,
        )

    return target


__all__ = ["create_capsule_scaffold"]
