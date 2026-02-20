from __future__ import annotations

import argparse
from typing import Sequence

from ...api.audit import build_audit_parser, run_audit


def build_parser() -> argparse.ArgumentParser:
    return build_audit_parser()


def main(argv: Sequence[str]) -> int:
    return run_audit(argv)
