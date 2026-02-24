"""
Compatibility module for built-in metric registrations.

Built-ins now live directly under `lelabo.metrics`.
This module is kept so existing imports keep working.
"""

from . import classification as _classification  # noqa: F401
from . import regression as _regression  # noqa: F401
