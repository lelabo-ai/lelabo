from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_META_SCHEMA_VERSION = "run_meta/v1"
RUN_SUMMARY_SCHEMA_VERSION = "run_summary/v1"
RUN_SEEDS_SCHEMA_VERSION = "run_seeds/v1"
RUN_METRICS_SCHEMA_VERSION = "run_metrics/v1"
RUN_CHECKPOINT_SCHEMA_VERSION = "run_checkpoint/v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_like(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_like(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_like(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def public_args_dict(args_or_mapping: Any) -> dict[str, Any]:
    if isinstance(args_or_mapping, Mapping):
        raw = dict(args_or_mapping)
    elif hasattr(args_or_mapping, "__dict__"):
        raw = dict(vars(args_or_mapping))
    else:
        return {}
    return {
        str(k): json_like(v)
        for k, v in raw.items()
        if not str(k).startswith("_")
    }


def build_persisted_summary(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    status: str,
    checkpoints_dir: str | None,
) -> dict[str, Any]:
    args = public_args_dict(summary.get("args", {}))
    artifacts = {
        "meta": "meta.json",
        "config": "resolved_config.yaml",
        "seeds": "seeds.json",
        "metrics": "metrics.jsonl",
        "checkpoints": checkpoints_dir,
    }

    run_block = {
        "task": args.get("task"),
        "dataset": args.get("dataset"),
        "model": args.get("model"),
        "algo": args.get("algo"),
    }

    if isinstance(summary.get("train"), Mapping):
        train_raw = dict(summary["train"])
        train_out = {
            key: json_like(train_raw[key])
            for key in ("final_epoch", "best", "runtime", "restoration", "run_metrics")
            if key in train_raw
        }
        out = {
            "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
            "run_id": str(run_id),
            "status": str(status),
            "run": run_block,
            "train": train_out,
            "eval": json_like(summary.get("eval", {})),
            "artifacts": artifacts,
        }
        if "robustness" in summary:
            out["robustness"] = json_like(summary["robustness"])
        return out

    if isinstance(summary.get("rl"), Mapping):
        out = {
            "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
            "run_id": str(run_id),
            "status": str(status),
            "run": {
                **run_block,
                "env": args.get("env"),
                "rl_algo": args.get("rl_algo"),
            },
            "rl": json_like(summary["rl"]),
            "artifacts": artifacts,
        }
        return out

    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "run_id": str(run_id),
        "status": str(status),
        "run": run_block,
        "artifacts": artifacts,
        "payload": json_like({k: v for k, v in summary.items() if str(k) != "args"}),
    }
