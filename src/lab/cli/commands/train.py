from __future__ import annotations

import sys
from typing import Sequence

from .._common import coerce_system_exit_code


def main(argv: Sequence[str]) -> int:
    from ...main import main as train_main

    prev_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *list(argv)]
    try:
        train_main()
    except SystemExit as exc:
        return coerce_system_exit_code(exc.code)
    finally:
        sys.argv = prev_argv
    return 0

