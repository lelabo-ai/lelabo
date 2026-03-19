"""Public capsule lifecycle API for scaffolding, packing, and storage."""

from .schema import CAPSULE_SCHEMA_VERSION, validate_manifest
from .collect import collect_capsule
from .pack import pack_capsule
from .install import compute_capsule_fingerprint, inspect_capsule_directory, install_capsule, install_capsule_from_directory
from .gitspace import add_capsule_to_gitspace, find_gitspace_root, init_gitspace, load_gitspace, resolve_gitspace_capsule
from .registry import list_capsules, get_capsule, default_capsules_dir
from .publish import available_targets, infer_workspace_target, list_configured_targets, save_configured_target, set_default_target
from .rerun import rerun_capsule
from .create import create_capsule_scaffold
from .remove import remove_capsule
from .restore import checkout_capsule
from .share import current_github_login, git_repo_root, share_capsule_github, share_gitspace_github
from .store import attach_capsule, stash_capsule

__all__ = [
    "CAPSULE_SCHEMA_VERSION",
    "validate_manifest",
    "collect_capsule",
    "pack_capsule",
    "install_capsule",
    "install_capsule_from_directory",
    "compute_capsule_fingerprint",
    "inspect_capsule_directory",
    "init_gitspace",
    "load_gitspace",
    "find_gitspace_root",
    "add_capsule_to_gitspace",
    "resolve_gitspace_capsule",
    "list_capsules",
    "get_capsule",
    "default_capsules_dir",
    "available_targets",
    "infer_workspace_target",
    "list_configured_targets",
    "save_configured_target",
    "set_default_target",
    "rerun_capsule",
    "create_capsule_scaffold",
    "remove_capsule",
    "stash_capsule",
    "attach_capsule",
    "checkout_capsule",
    "current_github_login",
    "git_repo_root",
    "share_gitspace_github",
    "share_capsule_github",
]
