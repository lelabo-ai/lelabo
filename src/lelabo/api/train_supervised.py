from __future__ import annotations

from argparse import Namespace
from typing import Any

from ..core.runners.supervised_runner import run_supervised
from ..core.utils.logger import RunLogger


def run_supervised_train(args: Namespace, logger: RunLogger) -> dict[str, Any]:
    return run_supervised(args, logger)


__all__ = ["run_supervised_train"]

