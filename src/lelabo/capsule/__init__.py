"""Public capsule lifecycle API for scaffolding, packing, and storage."""

from .schema import CAPSULE_SCHEMA_VERSION, validate_manifest
from .collect import collect_capsule
from .pack import pack_capsule
from .install import install_capsule
from .registry import list_capsules, get_capsule, default_capsules_dir
from .rerun import rerun_capsule
from .create import create_capsule_scaffold
from .remove import remove_capsule
from .restore import checkout_capsule
from .store import stash_capsule

__all__ = [
    "CAPSULE_SCHEMA_VERSION",
    "validate_manifest",
    "collect_capsule",
    "pack_capsule",
    "install_capsule",
    "list_capsules",
    "get_capsule",
    "default_capsules_dir",
    "rerun_capsule",
    "create_capsule_scaffold",
    "remove_capsule",
    "stash_capsule",
    "checkout_capsule",
]
