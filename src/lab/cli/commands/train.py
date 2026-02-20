from __future__ import annotations

from typing import Sequence

from ...api.train import run_train_from_argv


def main(argv: Sequence[str]) -> int:
    run_train_from_argv(list(argv))
    return 0
