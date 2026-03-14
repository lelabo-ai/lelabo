from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
logger_mod = importlib.import_module("lelabo.core.logger")
train_api = importlib.import_module("lelabo.cli.commands.train")


def test_run_logger_meta_lifecycle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_meta"
    logger = logger_mod.RunLogger(run_dir=run_dir, save_checkpoints=True)

    logger.write_meta({"dataset": "iris", "seed": 2}, task="supervised")
    meta_path = run_dir / "meta.json"
    assert meta_path.exists()

    started = json.loads(meta_path.read_text(encoding="utf-8"))
    assert started["schema_version"] == "run_meta/v1"
    assert started["status"] == "running"
    assert started["completed_at"] is None
    assert started["task"] == "supervised"
    assert started["checkpointing"]["enabled"] is True
    assert started["checkpointing"]["dir"] == "checkpoints"

    logger.finalize_meta("succeeded")
    ended = json.loads(meta_path.read_text(encoding="utf-8"))
    assert ended["status"] == "succeeded"
    assert ended["completed_at"] is not None


def test_supervised_run_writes_standard_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "quickstart_run"
    summary = train_api.run_train_from_argv(
        [
            "supervised",
            "--config",
            str(REPO_ROOT / "configs" / "train" / "supervised.quickstart.toml"),
            "--epochs",
            "1",
            "--display",
            "none",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert summary["args"]["dataset"] == "iris"

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    seeds = json.loads((run_dir / "seeds.json").read_text(encoding="utf-8"))
    persisted = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert meta["status"] == "succeeded"
    assert meta["summary_schema"] == "run_summary/v1"
    assert meta["metrics_schema"] == "run_metrics/v1"
    assert meta["checkpointing"]["enabled"] is False
    assert seeds["schema_version"] == "run_seeds/v1"
    assert seeds["seed"] == 2
    assert seeds["determinism"] == "relaxed"
    assert resolved["dataset"]["name"] == "iris"
    assert resolved["runtime"]["device"] in {"cpu", "cuda"}
    assert resolved["runtime"]["save_checkpoints"] is False
    assert persisted["schema_version"] == "run_summary/v1"
    assert persisted["status"] == "succeeded"
    assert "history" not in persisted["train"]
    assert persisted["artifacts"]["config"] == "resolved_config.yaml"
    assert persisted["artifacts"]["checkpoints"] is None
    assert not (run_dir / "checkpoints").exists()

    train_records = [rec for rec in metrics if rec.get("t") == "train"]
    eval_records = [rec for rec in metrics if rec.get("t") == "eval"]
    assert train_records
    assert eval_records
    assert train_records[0]["event"] == "epoch_end"
    assert train_records[0]["split"] == "train"
    assert "timestamp" in train_records[0]
    assert "num_samples" in train_records[0]
    assert "num_batches" in train_records[0]
    assert eval_records[0]["event"] == "eval_end"
    assert "timestamp" in eval_records[0]


def test_supervised_run_opt_in_checkpoints_writes_last_and_best(tmp_path: Path) -> None:
    run_dir = tmp_path / "ckpt_run"
    train_api.run_train_from_argv(
        [
            "supervised",
            "--config",
            str(REPO_ROOT / "configs" / "train" / "supervised.quickstart.toml"),
            "--epochs",
            "1",
            "--display",
            "none",
            "--run-dir",
            str(run_dir),
            "--save-checkpoints",
        ]
    )

    checkpoints = run_dir / "checkpoints"
    assert (checkpoints / "last.pt").exists()
    assert (checkpoints / "best.pt").exists()

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    persisted = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert meta["checkpointing"]["enabled"] is True
    assert meta["checkpointing"]["dir"] == "checkpoints"
    assert persisted["artifacts"]["checkpoints"] == "checkpoints"


def test_save_checkpoints_requires_run_dir() -> None:
    with pytest.raises(ValueError, match="run_dir"):
        train_api.run_train_from_argv(
            [
                "supervised",
                "--config",
                str(REPO_ROOT / "configs" / "train" / "supervised.quickstart.toml"),
                "--epochs",
                "1",
                "--display",
                "none",
                "--save-checkpoints",
            ]
        )
