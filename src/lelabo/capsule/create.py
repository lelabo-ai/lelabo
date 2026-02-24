from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .. import __version__ as LELABO_VERSION
from ..config.versioning import TRAIN_CONFIG_SCHEMA_VERSION
from .registry import add_capsule_entry, default_capsules_dir
from .schema import CAPSULE_SCHEMA_VERSION, validate_manifest

_CAPSULE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SCAFFOLD_DIRS = (
    "models",
    "update_rules",
    "datasets",
    "metrics",
    "optimizers",
    "schedulers",
    "configs",
    "runs",
)

_EXAMPLE_TEMPLATE_MAP = {
    "models/example.py": "models_example.py",
    "update_rules/example.py": "update_rules_example.py",
    "datasets/example.py": "datasets_example.py",
    "metrics/example.py": "metrics_example.py",
    "optimizers/example.py": "optimizers_example.py",
    "schedulers/example.py": "schedulers_example.py",
    "configs/README.md": "configs_readme.md",
    "configs/train.supervised.quickstart.toml": "configs_train_supervised_quickstart.toml",
    "configs/train.supervised.detailed.toml": "configs_train_supervised_detailed.toml",
    "configs/train.rl.detailed.toml": "configs_train_rl_detailed.toml",
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
        "- `optimizers/`: local optimizer builders\n"
        "- `schedulers/`: local learning-rate schedulers\n"
        "- `configs/`: experiment configs\n"
        "- `runs/`: local run artifacts\n"
        "\n"
        "Each folder contains starter templates you can adapt.\n"
        "\n"
        "## Local Registries\n\n"
        "When you run `lelabo` inside this capsule (or a subfolder), LeLabo auto-loads\n"
        "`models/*.py`, `update_rules/*.py`, `datasets/*.py`, `metrics/*.py`,\n"
        "`optimizers/*.py`, and `schedulers/*.py`.\n"
        "Use the standard decorators in these files:\n\n"
        "- `from lelabo.models.registry import register_model`\n"
        "- `from lelabo.update_rules.registry import register_update_rule`\n"
        "- `from lelabo.supervised.datasets.registry import register_dataset`\n"
        "- `from lelabo.metrics.registry import register_metric`\n"
        "- `from lelabo.optimizers import register_optimizer`\n"
        "- `from lelabo.schedulers import register_scheduler`\n"
    )


def _capsule_toml_template(name: str) -> str:
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        "[capsule]\n"
        f'name = "{name}"\n'
        f'created_at = "{created_at}"\n'
        f'lelabo_version = "{LELABO_VERSION}"\n'
        'format = "lelabo.capsule.scaffold.v1"\n'
    )


def _manifest_template(name: str, target: Path) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "capsule_id": name,
        "lelabo_version": LELABO_VERSION,
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
    pkg = resources.files("lelabo.capsule.templates")
    out: dict[str, str] = {}
    for rel_path, src_name in _EXAMPLE_TEMPLATE_MAP.items():
        src = pkg.joinpath(src_name)
        out[rel_path] = src.read_text(encoding="utf-8")
    return out


def _render_template_text(rel_path: str, content: str) -> str:
    rendered = str(content)
    if rel_path.startswith("configs/") and rel_path.endswith(".toml"):
        rendered = re.sub(
            r'(?m)^(\s*config_version\s*=\s*)"[^"]*"\s*$',
            rf'\1"{TRAIN_CONFIG_SCHEMA_VERSION}"',
            rendered,
        )
        rendered = re.sub(
            r'(?m)^(\s*lelabo_version\s*=\s*)"[^"]*"\s*$',
            rf'\1"{LELABO_VERSION}"',
            rendered,
        )
    return rendered


def _template_files(name: str) -> dict[str, str]:
    templates = {
        "README.md": _readme_template(name),
        "capsule.toml": _capsule_toml_template(name),
        "models/__init__.py": '"""Custom models for this capsule."""\n',
        "update_rules/__init__.py": '"""Custom update rules for this capsule."""\n',
        "datasets/__init__.py": '"""Custom datasets for this capsule."""\n',
        "metrics/__init__.py": '"""Custom metrics for this capsule."""\n',
        "optimizers/__init__.py": '"""Custom optimizers for this capsule."""\n',
        "schedulers/__init__.py": '"""Custom schedulers for this capsule."""\n',
    }
    templates.update(
        {
            rel_path: _render_template_text(rel_path, content)
            for rel_path, content in _load_example_templates().items()
        }
    )
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
