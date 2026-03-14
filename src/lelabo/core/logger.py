from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

import yaml

from .run_artifacts import (
    RUN_CHECKPOINT_SCHEMA_VERSION,
    RUN_META_SCHEMA_VERSION,
    RUN_METRICS_SCHEMA_VERSION,
    RUN_SEEDS_SCHEMA_VERSION,
    RUN_SUMMARY_SCHEMA_VERSION,
    build_persisted_summary,
    json_like,
    public_args_dict,
    utc_now_iso,
)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _to_cpu_state(raw: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(raw):
        return raw.detach().cpu().clone()
    if isinstance(raw, Mapping):
        return {str(k): _to_cpu_state(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_to_cpu_state(v) for v in raw]
    if isinstance(raw, tuple):
        return tuple(_to_cpu_state(v) for v in raw)
    return raw


def _checkpoint_mode_from_name(name: str) -> str:
    token = str(name).strip().lower()
    if any(item in token for item in ("acc", "f1", "precision", "recall", "r2", "auc")):
        return "max"
    return "min"


@dataclass
class RunLogger:
    run_dir: Optional[Path] = None
    save_checkpoints: bool = False
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def __post_init__(self) -> None:
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path = self.run_dir / "metrics.jsonl"
            self.meta_path = self.run_dir / "meta.json"
            self.summary_path = self.run_dir / "summary.json"
            self.config_path = self.run_dir / "resolved_config.yaml"
            self.seeds_path = self.run_dir / "seeds.json"
            self.checkpoints_path = self.run_dir / "checkpoints" if self.save_checkpoints else None
        else:
            self.metrics_path = None
            self.meta_path = None
            self.summary_path = None
            self.config_path = None
            self.seeds_path = None
            self.checkpoints_path = None
        self._meta_payload: dict[str, Any] | None = None
        self._best_monitor_name: str | None = None
        self._best_monitor_mode: str | None = None
        self._best_monitor_value: float | None = None
        self._best_checkpoint_payload: dict[str, Any] | None = None

    def log(self, record: Mapping[str, Any]) -> None:
        if self.metrics_path is None:
            return
        payload = {str(k): json_like(v) for k, v in dict(record).items()}
        payload.setdefault("timestamp", utc_now_iso())
        append_jsonl(self.metrics_path, payload)

    def write_meta(self, args: Mapping[str, Any] | Any, *, task: str) -> None:
        if self.meta_path is None:
            return
        public_args = public_args_dict(args)
        started_at = utc_now_iso()
        payload = {
            "schema_version": RUN_META_SCHEMA_VERSION,
            "run_id": str(self.run_id),
            "status": "running",
            "timestamp": started_at,
            "started_at": started_at,
            "completed_at": None,
            "git_commit": safe_git_commit(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "argv": list(map(str, os.sys.argv)),
            "cwd": str(Path.cwd().resolve()),
            "task": str(task),
            "metrics_schema": RUN_METRICS_SCHEMA_VERSION,
            "summary_schema": RUN_SUMMARY_SCHEMA_VERSION,
            "config_artifact": "resolved_config.yaml",
            "args_note": "Args are kept for compatibility. Use resolved_config.yaml as the canonical config artifact.",
            "checkpointing": {
                "enabled": bool(self.checkpoints_path is not None),
                "dir": None if self.checkpoints_path is None else "checkpoints",
            },
            "args": public_args,
        }
        self._meta_payload = payload
        write_json(self.meta_path, payload)

    def finalize_meta(self, status: str, *, error: str | None = None) -> None:
        if self.meta_path is None:
            return
        payload = dict(self._meta_payload or {})
        if not payload:
            completed_at = utc_now_iso()
            payload = {
                "schema_version": RUN_META_SCHEMA_VERSION,
                "run_id": str(self.run_id),
                "status": str(status),
                "started_at": completed_at,
                "completed_at": completed_at,
            }
        else:
            payload["status"] = str(status)
            payload["completed_at"] = utc_now_iso()
        if error is not None:
            payload["error"] = str(error)
        self._meta_payload = payload
        write_json(self.meta_path, payload)

    def write_resolved_config(self, resolved_config: Mapping[str, Any] | Any) -> None:
        if self.config_path is None:
            return
        payload = json_like(resolved_config)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def write_seeds(self, seed_state: Any) -> None:
        if self.seeds_path is None:
            return
        payload = {
            "schema_version": RUN_SEEDS_SCHEMA_VERSION,
            "seed": int(getattr(seed_state, "seed", 0)),
            "determinism": str(getattr(seed_state, "mode", "relaxed")),
            "deterministic_algorithms": bool(getattr(seed_state, "deterministic_algorithms", False)),
            "cudnn_deterministic": getattr(seed_state, "cudnn_deterministic", None),
            "cudnn_benchmark": getattr(seed_state, "cudnn_benchmark", None),
            "cublas_workspace_config": getattr(seed_state, "cublas_workspace_config", None),
        }
        write_json(self.seeds_path, payload)

    def write_summary(self, summary: Mapping[str, Any], *, status: str) -> None:
        if self.summary_path is None:
            return
        payload = build_persisted_summary(
            summary,
            run_id=self.run_id,
            status=status,
            checkpoints_dir=(None if self.checkpoints_path is None else "checkpoints"),
        )
        write_json(self.summary_path, payload)

    def build_checkpoint_payload(
        self,
        *,
        task: str,
        model: Any,
        learner: Any | None = None,
        optimizer: Any | None = None,
        schedulers: list[Any] | None = None,
        epoch: int | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": RUN_CHECKPOINT_SCHEMA_VERSION,
            "run_id": str(self.run_id),
            "task": str(task),
            "epoch": None if epoch is None else int(epoch),
        }
        state_dict = getattr(model, "state_dict", None)
        if callable(state_dict):
            payload["model"] = _to_cpu_state(state_dict())
        learner_state = getattr(learner, "state_dict", None)
        if callable(learner_state):
            payload["learner"] = _to_cpu_state(learner_state())
        if optimizer is not None:
            payload["optimizer"] = _to_cpu_state(optimizer.state_dict())
        if schedulers:
            payload["schedulers"] = [
                _to_cpu_state(sched.state_dict()) if callable(getattr(sched, "state_dict", None)) else None
                for sched in list(schedulers)
            ]
        if meta:
            payload["meta"] = json_like(meta)
        return payload

    def write_checkpoint(self, name: str, payload: Mapping[str, Any]) -> None:
        if self.checkpoints_path is None:
            return
        import torch

        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        torch.save(dict(payload), self.checkpoints_path / f"{name}.pt")

    def _resolve_checkpoint_monitor(self, trainer: Any, epoch_record: Any) -> tuple[str, str] | tuple[None, None]:
        if self._best_monitor_name is not None and self._best_monitor_mode is not None:
            return self._best_monitor_name, self._best_monitor_mode

        from .callbacks import EarlyStopping

        for callback in list(getattr(trainer, "callbacks", []) or []):
            if isinstance(callback, EarlyStopping):
                status = callback.status()
                self._best_monitor_name = str(status.name)
                self._best_monitor_mode = str(status.mode)
                return self._best_monitor_name, self._best_monitor_mode

        for scheduler in list(getattr(trainer, "schedulers", []) or []):
            monitor_name = str(getattr(scheduler, "monitor", "")).strip()
            if not monitor_name:
                continue
            raw_mode = getattr(getattr(scheduler, "scheduler", None), "mode", None)
            if str(raw_mode).strip().lower() in {"min", "max"}:
                mode = str(raw_mode).strip().lower()
            else:
                mode = _checkpoint_mode_from_name(monitor_name)
            self._best_monitor_name = monitor_name
            self._best_monitor_mode = mode
            return self._best_monitor_name, self._best_monitor_mode

        if getattr(epoch_record, "val", None) is not None:
            self._best_monitor_name = "val.loss"
            self._best_monitor_mode = "min"
        else:
            self._best_monitor_name = "train.loss"
            self._best_monitor_mode = "min"
        return self._best_monitor_name, self._best_monitor_mode

    def on_fit_start(self, trainer: Any, state: Any | None = None) -> None:
        _ = (trainer, state)
        self._best_monitor_name = None
        self._best_monitor_mode = None
        self._best_monitor_value = None
        self._best_checkpoint_payload = None

    def on_epoch_end(self, trainer: Any, epoch_record: Any, state: Any | None = None) -> None:
        _ = state
        if self.checkpoints_path is None:
            return
        monitor_name, monitor_mode = self._resolve_checkpoint_monitor(trainer, epoch_record)
        if not monitor_name or not monitor_mode:
            return
        value = epoch_record.to_log_values().get(monitor_name)
        if not isinstance(value, (int, float)):
            return
        current_value = float(value)
        improved = self._best_monitor_value is None
        if self._best_monitor_value is not None:
            if monitor_mode == "max":
                improved = current_value > float(self._best_monitor_value)
            else:
                improved = current_value < float(self._best_monitor_value)
        if not improved:
            return
        self._best_monitor_value = current_value
        self._best_checkpoint_payload = self.build_checkpoint_payload(
            task="supervised",
            model=trainer.model,
            learner=trainer.learner,
            optimizer=getattr(trainer.learner, "optimizer", None),
            schedulers=list(getattr(trainer, "schedulers", []) or []),
            epoch=int(epoch_record.epoch),
            meta={
                "kind": "best",
                "monitor_name": monitor_name,
                "monitor_mode": monitor_mode,
                "monitor_value": current_value,
            },
        )

    def on_fit_end(self, trainer: Any, fit_result: Any, state: Any | None = None) -> None:
        _ = state
        if self.checkpoints_path is None:
            return
        last_payload = self.build_checkpoint_payload(
            task="supervised",
            model=trainer.model,
            learner=trainer.learner,
            optimizer=getattr(trainer.learner, "optimizer", None),
            schedulers=list(getattr(trainer, "schedulers", []) or []),
            epoch=int(getattr(getattr(fit_result, "final_epoch", None), "epoch", 0) or 0),
            meta={
                "kind": "last",
                "best_epoch": getattr(getattr(fit_result, "best", None), "epoch", None),
                "restored_on_train_end": bool(getattr(getattr(fit_result, "restoration", None), "restored_on_train_end", False)),
            },
        )
        self.write_checkpoint("last", last_payload)
        if self._best_checkpoint_payload is not None:
            self.write_checkpoint("best", self._best_checkpoint_payload)
