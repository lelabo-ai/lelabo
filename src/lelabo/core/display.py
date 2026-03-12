from __future__ import annotations

from typing import Any


VALID_DISPLAY_MODES = ("none", "compact", "rich")


def normalize_display_mode(raw: Any, *, where: str) -> str:
    token = str(raw if raw is not None else "").strip().lower()
    if token in VALID_DISPLAY_MODES:
        return token
    raise ValueError(f"{where} must be one of: none, compact, rich.")
