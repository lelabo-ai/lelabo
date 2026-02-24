"""Minimal internal API layer used by the CLI."""

from .audit import build_audit_parser, run_audit
from .capsule import (
    create_capsule,
    install_capsule,
    list_capsules,
    pack_capsule,
    remove_capsule,
    restore_capsule,
    rerun_capsule,
    show_capsule,
    store_capsule,
)
from .train import build_train_parser, parse_train_args, run_train, run_train_from_argv

__all__ = [
    "build_audit_parser",
    "build_train_parser",
    "parse_train_args",
    "run_audit",
    "run_train",
    "run_train_from_argv",
    "create_capsule",
    "pack_capsule",
    "install_capsule",
    "store_capsule",
    "list_capsules",
    "show_capsule",
    "rerun_capsule",
    "remove_capsule",
    "restore_capsule",
]
