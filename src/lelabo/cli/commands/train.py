from __future__ import annotations

from typing import Sequence

from ...api.train import main as run_train_main


def main(argv: Sequence[str]) -> int:
    return int(run_train_main(list(argv)))
