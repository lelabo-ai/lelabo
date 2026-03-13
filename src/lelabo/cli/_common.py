from __future__ import annotations

import sys


CLI_USER_ERROR_TYPES = (
    ValueError,
    TypeError,
    ImportError,
    FileNotFoundError,
    RuntimeError,
)


def is_help_token(raw: str) -> bool:
    return str(raw).strip().lower() in {"-h", "--help", "help"}


def coerce_system_exit_code(code: object) -> int:
    if isinstance(code, int):
        return code
    if code is None:
        return 0

    msg = str(code).strip()
    if msg:
        print(msg, file=sys.stderr)
    return 1
