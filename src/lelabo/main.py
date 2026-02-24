from __future__ import annotations

from typing import Sequence

from .api.train import main as run_train_cli


def main(argv: Sequence[str] | None = None) -> int:
    return int(run_train_cli(argv))


if __name__ == "__main__":
    raise SystemExit(main())
