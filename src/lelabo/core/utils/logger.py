# lab/core/logger.py
from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


@dataclass
class RunLogger:
    run_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path = self.run_dir / "metrics.jsonl"
        else:
            self.metrics_path = None

    def log(self, record: Dict[str, Any]) -> None:
        if self.metrics_path is None:
            return
        append_jsonl(self.metrics_path, record)

    def write_meta(self, args: Dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": safe_git_commit(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "argv": list(map(str, os.sys.argv)),
            "args": args,
        }
        write_json(self.run_dir / "meta.json", meta)

    def write_summary(self, summary: Dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        write_json(self.run_dir / "summary.json", summary)
