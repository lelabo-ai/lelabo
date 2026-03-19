"""Shared human-facing terminal UI helpers for LeLabo CLI commands."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


_STATUS_LABELS = {
    "success": "Success",
    "info": "Info",
    "warning": "Warning",
    "next": "Next",
}


def _normalize_value(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text or "-"


def print_status(kind: str, message: str) -> None:
    """Print a short one-line status message."""
    label = _STATUS_LABELS.get(str(kind).strip().lower(), "Info")
    print(f"{label}: {str(message).strip()}")


def print_block(title: str, rows: Sequence[tuple[str, Any]]) -> None:
    """Print a compact title + key/value block."""
    print(str(title).strip())
    for key, value in rows:
        print(f"{key}: {_normalize_value(value)}")


def print_kv_list(rows: Sequence[tuple[str, Any]]) -> None:
    """Print only key/value rows without a block title."""
    for key, value in rows:
        print(f"{key}: {_normalize_value(value)}")


def print_list_block(title: str, items: Iterable[str], *, empty_message: str = "- (none)") -> None:
    """Print a titled bullet list."""
    print(str(title).strip())
    materialized = [str(item).strip() for item in items if str(item).strip()]
    if not materialized:
        print(empty_message)
        return
    for item in materialized:
        print(f"- {item}")


def print_next_steps(items: Sequence[str], *, title: str = "Next") -> None:
    """Print a short next-steps section."""
    print(f"{title}:")
    for item in items:
        token = str(item).strip()
        if token:
            print(f"- {token}")


def build_prompt_style():
    """Return the shared prompt_toolkit style used by interactive flows."""
    try:
        from prompt_toolkit.styles import Style
    except Exception:
        return None

    return Style.from_dict(
        {
            "label": "",
            "button": "",
            "button.focused": "reverse bold",
            "checkbox": "",
            "checkbox-selected": "bold",
            "checkbox-list": "",
            "checkbox-list.current": "reverse",
            "error": "#b42318 bold",
            "muted": "#6b7280",
            "accent": "bold",
        }
    )


__all__ = [
    "build_prompt_style",
    "print_block",
    "print_kv_list",
    "print_list_block",
    "print_next_steps",
    "print_status",
]
