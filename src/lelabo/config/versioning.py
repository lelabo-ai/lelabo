"""Version helpers for train-config schemas and LeLabo compatibility markers."""

from __future__ import annotations

try:
    from .. import __version__ as LELABO_VERSION
except Exception:  # pragma: no cover - fallback for namespace-stubbed test environments.
    LELABO_VERSION = "0.0.0"


TRAIN_CONFIG_SCHEMA_VERSION = "1.0"
_AUTO_TOKENS = {"", "auto", "current", "latest"}


def resolve_config_version(raw: object) -> str:
    text = str(raw or "").strip()
    if text.lower() in _AUTO_TOKENS:
        return TRAIN_CONFIG_SCHEMA_VERSION
    return text


def resolve_lelabo_version(raw: object) -> str:
    text = str(raw or "").strip()
    if text.lower() in _AUTO_TOKENS:
        return LELABO_VERSION
    return text
